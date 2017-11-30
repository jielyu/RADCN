# encoding:utf-8

# System libraries
import os
import sys

import pickle
import numpy as np
import tensorflow as tf

from ObjectDetectionRAM import Config as ODRAMConfig
from ObjectDetectionRAM import ObjectDetectionRAM as ODRAM
from Reward.ClassificationAction import ClassificationAction as ClaAction
from Reward.ClassificationAction import Config as ClaConfig
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset
from src.tools.Utils import Utils
from RAMEvaluator import RAMEvaluator

class Config(ODRAMConfig):

    def __init__(self,
                 isRandomInitial=True,
                 initialMinValue=-0.3,
                 initialMaxValue=0.3,
                 inputShape=[56, 56, 1],
                 numCategory=10,
                 isAddContext=False,
                 nScale=3,
                 scaleFactor=2.0,
                 minScaleSize=8,
                 targetSize=16,
                 featureDim=256,
                 focusedCoordinateNum=2,
                 objectDim = 5,
                 maxObjectNum=1,
                 batchSize=64,
                 fusionMethod='plus',
                 maxTimeStep=10,
                 lstmLayers=3,
                 isTrain=False,
                 keepProb=1.0,
                 monteCarloSample=1,
                 isSamplePoint=True,
                 samplingStd=0.2,
                 maxGradNorm=500,
                 startLearningRate=1e-4,
                 minLearningRate=1e-5,
                 learningStepFactor=0.97,
                 maxTrainEpoch=40,
                 modelFileName='CDRAM.ckpt',
                 homeDir='/home/jielyu/Workspace/Python/RADCN',
                 modelRelativeDir='output/RAM/CDRAM',
                 initialModelDir='',
                 logFileName='CDTask.log'):
        super(Config, self).__init__(
            isRandomInitial=isRandomInitial,
            initialMinValue=initialMinValue,
            initialMaxValue=initialMaxValue,
            inputShape=inputShape,
            isAddContext=isAddContext,
            nScale=nScale,
            scaleFactor=scaleFactor,
            minScaleSize=minScaleSize,
            targetSize=targetSize,
            featureDim=featureDim,
            focusedCoordinateNum=focusedCoordinateNum,
            objectDim=objectDim,
            maxObjectNum=maxObjectNum,
            batchSize=batchSize,
            fusionMethod=fusionMethod,
            maxTimeStep=maxTimeStep,
            lstmLayers=lstmLayers,
            isTrain=isTrain,
            keepProb=keepProb,
            monteCarloSample=monteCarloSample,
            isSamplePoint=isSamplePoint,
            samplingStd=samplingStd,
            maxGradNorm=maxGradNorm,
            startLearningRate=startLearningRate,
            minLearningRate=minLearningRate,
            learningStepFactor=learningStepFactor,
            maxTrainEpoch=maxTrainEpoch,
            modelFileName=modelFileName,
            homeDir=homeDir,
            modelRelativeDir=modelRelativeDir,
            initialModelDir=initialModelDir,
            logFileName=logFileName)
        self.numCategory = numCategory

    def saveTo(self, fid):
        if not isinstance(fid, file):
            raise TypeError('Not file object')
        super(Config, self).saveTo(fid=fid)
        fid.write('\r\n')
        fid.write('numCategory = %s\r\n' % (self.numCategory))
        fid.write('\r\n\r\n')


class ClassificationDetectionRAM(ODRAM):

    def __init__(self, config=Config()):
        super(ClassificationDetectionRAM, self).__init__(config=config)

        claConf = ClaConfig(featureDim=config.featureDim,
                            numCategory=config.numCategory,
                            isTrain=config.isTrain,
                            keepProb=config.keepProb,
                            isDropout=True)
        self.classifier = ClaAction(config=claConf)

        self.labels = 0

    def buildCoreGraph(self, images, points,
                       name='CDRAM_timestep', reuse=False):
        # Build the graph at a time step
        with tf.variable_scope(name, reuse=reuse):
            # Extract multi-scale regions
            multiScaleImages = self.eyeLikeCapture.buildGraph(
                images=images,
                points=points)

            # Extract CNN representation
            cnnFeat = []
            for i in range(0, self.config.nScale):
                name = 'EyeLikeScale_' + str(i)
                with tf.variable_scope(name):
                    cnnFeat.append(
                        self.cnnExtractor[i].buildGraph(
                            multiScaleImages[i], reuse=reuse)
                    )

            # Concat multi-scale CNN representations
            concatedCNNFeat = tf.concat(axis=1, values=cnnFeat)

            # Fusion
            fusionFeat = self.alFusion.buildGraph(concatedCNNFeat, points, reuse=reuse)

            # Extract LSTM representation
            lstmFeat = self.lstmExtractor.buildGraph(fusionFeat, reuse=reuse)

            # Predict the next focused points
            focusedPoints = self.fpPredictor.buildGraph(lstmFeat, reuse=reuse)

            # Generate boxes and scores of objects
            objs = self.objectDetectionReward.buildGraph(lstmFeat, reuse=reuse)

            claProb = self.classifier.buildGraph(lstmFeat, reuse=reuse)

            # Return
            return focusedPoints, objs, claProb

    def buildGraph(self, name='CDRAM'):
        # Define the input interface
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None,
                                            self.config.inputShape[0],
                                            self.config.inputShape[1],
                                            self.config.inputShape[2]])

        self.objects = tf.placeholder(dtype=tf.float32,
                                      shape=[None,
                                             self.config.maxObjectNum,
                                             self.config.objectDim - 1])

        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, self.config.numCategory])

        images = self.images
        with tf.variable_scope(name):
            # Get the number of samples
            numSamples = tf.shape(images)[0]
            dimPoint = self.config.focusedCoordinateNum
            # Initialize the initial focusing points and object list
            if self.config.isRandomInitial:
                points = tf.random_uniform(shape=[numSamples,
                                                  dimPoint],
                                           minval=self.config.initialMinValue,
                                           maxval=self.config.initialMaxValue,
                                           dtype=tf.float32,
                                           name='initialPoints')
            else:
                points = tf.random_uniform(shape=[numSamples,
                                                  dimPoint],
                                           minval=-1e-6,
                                           maxval=1e-6,
                                           dtype=tf.float32,
                                           name='initialPoints')
            yxList = []
            hwList = []
            scoreList = []
            claList = []
            preLabelsList = []
            pointList = []
            pointMeanList = []
            pointList.append(points)
            pointMeanList.append(points)
            # Create RNN
            for i in range(0, self.config.maxTimeStep):
                # Loop at each time step
                if i == 0:
                    r_points, objs, cla = \
                        self.buildCoreGraph(
                            images=images,
                            points=points)
                else:
                    # Parameters sharing
                    r_points, objs, cla = \
                        self.buildCoreGraph(
                            images=images,
                            points=points,
                            reuse=True)
                # Obtain absolute coordinate and set range
                # points_abs = points + r_points
                points_abs = r_points
                points_mean = tf.clip_by_value(points_abs, -1.0, 1.0)
                # points_mean = tf.stop_gradient(points_mean)
                if self.config.isSamplePoint:
                    points = points_mean + tf.random_normal(
                        shape=[tf.shape(images)[0],
                               self.config.focusedCoordinateNum],
                        stddev=self.config.samplingStd)
                    points = tf.clip_by_value(points, -1.0, 1.0)
                else:
                    points = points_mean

                points = tf.stop_gradient(points)

                if len(objs) != 3:
                    raise ValueError('Wrong Object interface')
                yxList.append(objs[0])
                hwList.append(objs[1])
                scoreList.append(objs[2])
                claList.append(cla)
                preLabelsList.append(tf.argmax(cla, 1))
                pointList.append(points)
                pointMeanList.append(points_mean)

            # Get the object list of a sequence
            self.objList = [yxList, hwList, scoreList]
            self.claList = claList
            self.preLabelsList = preLabelsList
            self.pointList = pointList
            self.pointMeanList = pointMeanList
            self.preLabels = tf.argmax(self.claList[-1], 1)

    def buildLossGraph(self, name='LossFunction'):

        preYXList = self.objList[0]
        preHWList = self.objList[1]
        preScoreList = self.objList[2]
        preClaList = self.claList
        pointList = self.pointList
        pointMeanList = self.pointMeanList
        gtObjs = self.objects
        gtCla = self.labels
        loss = 0
        overlap = 0
        with tf.variable_scope(name):
            if len(preYXList) != self.config.maxTimeStep:
                raise ValueError('Not RNN output')

            # Get predicted baselines
            baselines = tf.stack(preScoreList)  # [timestep, batchsize]
            baselines = tf.transpose(baselines)  # [batchsize, timestep]

            # Distance square of predicted and ground truth objects
            dsList = self.buildDistanceSqareGraph(preYXList=preYXList,
                                                  preHWList=preHWList,
                                                  gtYXHW=gtObjs)
            ls = tf.reduce_mean(dsList[-1])

            # Compute overlap rate and mean square root of center point
            msrList, overlapList = \
                self.buildOverlapGraph(preYXList=preYXList,
                                       preHWList=preHWList,
                                       gtYXHW=gtObjs)
            msr = msrList[-1]  # [batchsize,]
            overlap = overlapList[-1]  # [batchsize,]

            # Compute cross-entropy
            ceList, accList, rwdList = \
                self.buildEntropyGraph(preClaList=preClaList,
                                       gtCla=gtCla)
            ce = ceList[-1]
            acc = accList[-1]
            rwd = rwdList[-1]

            # Reward
            reward = overlap
            # reward = (rwd + overlap)/2.0
            # reward = rwd
            rewards = tf.expand_dims(reward, 1)  # [batchsize, timestep]
            rewards = tf.tile(rewards,
                              (1, self.config.maxTimeStep))

            bias = rewards - tf.stop_gradient(baselines)
            baselines_mse = tf.reduce_mean(
                tf.square(rewards - baselines)
            )

            # Compute log likelihood of points
            logll = self.buildLogLikelihoodGraph(pointList=pointList,
                                                 pointMeanList=pointMeanList)
            logll = tf.reduce_mean(logll * bias)
            # logll = tf.reduce_mean(logll * rewards)
            # logll = tf.reduce_mean(logll*tf.square(rewards - baselines))

            # Compute attention distance
            ads = self.buildAttentionDistanceGraph(pointMeanList, gtObjs)
            ads = tf.reduce_mean(ads[-1])

            # Loss function
            overlap = tf.reduce_mean(overlap) + 1e-6
            msr = tf.reduce_mean(msr)
            loss = -logll + ls + baselines_mse - tf.log(overlap) + ce + ads
            # loss = -logll + ce + baselines_mse
            # loss = -logll + ls + baselines_mse - tf.log(overlap)

        # The overlap of the last step
        reward = tf.reduce_mean(reward)
        self.loss = loss
        self.overlap = overlap
        self.msr = msr
        self.reward = reward
        self.baseline_mse = baselines_mse
        self.ls = ls
        self.ads = ads
        self.logll = logll
        self.ce = ce
        self.acc = acc

    def trainModel(self, modelDir='', sess=None):

        # Check training flag
        if not self.config.isTrain:
            raise ValueError('Not Training Configuration')
        # Check modelDir
        if not os.path.isdir(modelDir):
            modelDir = self.config.modelDir

        if sess is None:
            sess = self.createSession()
            # Load initialize model
        if os.path.isdir(self.config.initialModelDir):
            Utils.loadModel(sess=sess,
                            modelDir=self.config.initialModelDir,
                            modelFileName=self.config.modelFileName)
        # Train model
        self.printLog('start to train RAM')

        t_str = ': trainTimes=%s' % (self.config.maxTrainTimes)
        self.printLog(t_str)
        t_str = ': dispTimes=%s' % (self.config.minDispStep)
        self.printLog(t_str)

        lossTrainList = []
        overlapTrainList = []
        accTrainList = []
        lossTestList = []
        overlapTestList = []
        accTestList = []
        for i in range(0, self.config.maxTrainTimes):

            epochCnt = i // self.config.minDispStep
            # Obtain a batch of data
            images, bbox, labels = \
                self.dataset.trainset.getNextBatchWithLabels()
            bbox = Utils.convertToYXHW(bbox)
            images, bbox = Utils.normalizeImagesAndBbox(images, bbox)
            images = \
                np.tile(images, [self.config.monteCarloSample, 1, 1, 1])
            bbox = np.tile(bbox, [self.config.monteCarloSample, 1, 1])
            labels = np.tile(labels, [self.config.monteCarloSample, 1])
            feed_dict = {self.images: images,
                         self.objects: bbox,
                         self.labels:labels}
            # Train
            sess.run(self.trainOp, feed_dict=feed_dict)

            # Test
            if i == 0 \
                    or i % self.config.minDispStep == \
                                    self.config.minDispStep-1 \
                    or self.config.isDispAlways:
                # Test on trainset
                loss, overlap, msr, reward, bmse, ls, \
                ads, logll, ce, acc, lr = \
                    sess.run((self.loss, self.overlap, self.msr,
                              self.reward, self.baseline_mse,
                              self.ls, self.ads, self.logll,
                              self.ce, self.acc, self.learningRate),
                             feed_dict=feed_dict)
                # Display information
                t_str = 'step=%d ' % (i) + \
                        'epoch=%d/%d ' % (epochCnt,
                                          self.config.maxTrainEpoch) + \
                        'loss=%f ' % (loss) + \
                        'ol=%f ' % (overlap) + \
                        'msr=%f ' % (msr) + \
                        'rwd=%f ' % (reward) + \
                        'bmse=%f ' % (bmse) + \
                        'ls=%f ' % (ls) + \
                        'ads=%f ' % (ads) + \
                        'logll=%f ' % (logll) + \
                        'ce=%f ' %(ce) + \
                        'acc=%f '%(acc) + \
                        'lr=%f ' % (lr)
                lossTrainList.append(loss)
                overlapTrainList.append(overlap)
                accTrainList.append(acc)
                self.printLog(t_str)

                # Test and save
                if i == 0 \
                        or i % self.config.minDispStep == \
                                        self.config.minDispStep-1:
                    # Save model
                    Utils.saveModel(sess=sess,
                                    modelDir=modelDir,
                                    modelFileName=self.config.modelFileName)
                    # Test on testset
                    teLoss, teOverlap, teAcc = self.testModel(sess=sess)

                    lossTestList.append(teLoss)
                    overlapTestList.append(teOverlap)
                    accTestList.append(teAcc)
        self.printLog('Train completely')

        # Draw Training curve
        self.saveTrainingCurve(scalarList=lossTrainList,
                               name='train_loss')
        self.saveTrainingCurve(scalarList=overlapTrainList,
                               name='train_overlap')
        self.saveTrainingCurve(scalarList=accTrainList,
                               name='train_acc')
        self.saveTrainingCurve(scalarList=lossTestList,
                               name='test_loss')
        self.saveTrainingCurve(scalarList=overlapTestList,
                               name='test_overlap')
        self.saveTrainingCurve(scalarList=accTestList,
                               name='test_acc')

    def testModel(self,
                  modelDir='',
                  testset=None,
                  isSaveTrack=False,
                  isSaveData=False,
                  saveDir=None,
                  sess=None):
        print 'testing on the testset ... '
        # Check model direction
        if not os.path.isdir(modelDir):
            modelDir = self.config.modelDir

        if sess is None:
            sess = self.createSession()
            # Load model
            if not self.config.isTrain:
                Utils.loadModel(sess=sess,
                                modelDir=modelDir,
                                modelFileName=self.config.modelFileName)

        # Test on testset
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumOverlap = 0
        sumMsr = 0
        sumAcc = 0
        testSampleNum = \
            (numTestSample // self.config.batchSize) + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox, t_labels = \
                self.dataset.testset.getNextBatchWithLabels()
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.objects: t_bbox,
                         self.labels: t_labels}
            # Obtain accuracy
            # t_loss, t_overlap, t_msr = \
            #     sess.run((self.loss, self.overlap, self.msr),
            #              feed_dict=feed_dict)
            loss, overlap, msr, reward, bmse, ls, logll, ce, acc = \
                sess.run((self.loss, self.overlap, self.msr,
                          self.reward, self.baseline_mse,
                          self.ls, self.logll, self.ce, self.acc),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            sumMsr += msr / testSampleNum
            sumAcc += acc / testSampleNum

            if isSaveTrack:
                for k in range(0, 3):
                    # Obtain accuracy
                    t_overlap, t_msr, pointList, objList, preLabels = \
                        sess.run((self.overlap, self.msr,
                                  self.pointList, self.objList,
                                  self.preLabels),
                                 feed_dict=feed_dict)
                    if saveDir is None:
                        saveDir = os.path.join(self.config.homeDir,
                                               'output/CDRAM/test')
                    t_dir = os.path.join(saveDir,
                                         'batch_' + str(j) + '_' + str(k))
                    gtLabels = np.argmax(t_labels, axis=1)
                    self.saveRecurrentTrack(saveDir=t_dir,
                                            images=t_images,
                                            pointsList=pointList,
                                            YXList=objList[0],
                                            HWList=objList[1],
                                            scoreList=objList[2],
                                            preClaName=preLabels,
                                            gtBbox=t_bbox,
                                            gtClaName=gtLabels,
                                            isSaveData=isSaveData)
        t_str = 'testing overlap=%f ' % (sumOverlap) + \
                'msr=%f ' % (sumMsr) + \
                'acc=%f ' % (sumAcc)
        self.printLog(t_str)

        return sumLoss, sumOverlap, sumAcc

    def evaluate(self,
                 modelDir='',
                 sess=None,
                 saveDir=None,
                 saveMaxNum=100,
                 isSaveSeq=False,
                 ext='png'):
        print 'evaluating on the testset ... '
        # Check model direction
        if not os.path.isdir(modelDir):
            modelDir = self.config.modelDir

        if sess is None:
            sess = self.createSession()
            # Load model
            if not self.config.isTrain:
                Utils.loadModel(sess=sess,
                                modelDir=modelDir,
                                modelFileName=self.config.modelFileName)

        # Test on testset
        imagesList = []
        gtLabelsList = []
        gtBboxesList = []
        pointsListList = []
        preYXListList = []
        preHWListList = []
        preScoreListList = []
        preLabelsListList = []
        # preLabelsList = []

        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumOverlap = 0
        sumMsr = 0
        sumAcc = 0
        testSampleNum = \
            (numTestSample // self.config.batchSize) + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox, t_labels = \
                self.dataset.testset.getNextBatchWithLabels()
            gtLabels = np.argmax(t_labels, axis=1)
            imagesList.append(t_images)
            gtBboxesList.append(t_bbox)
            gtLabelsList.append(gtLabels)
            # Convert to standard mode
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.objects: t_bbox,
                         self.labels: t_labels}
            # Run graph
            loss, overlap, msr, \
            reward, bmse, \
            ls, logll, ce, \
            acc, pointList, \
            objList, preLabelsList, preLabels = \
                sess.run((self.loss, self.overlap, self.msr,
                          self.reward, self.baseline_mse,
                          self.ls, self.logll, self.ce,
                          self.acc, self.pointList,
                          self.objList, self.preLabelsList, self.preLabels),
                         feed_dict=feed_dict)
            pointsListList.append(pointList)
            preYXListList.append(objList[0])
            preHWListList.append(objList[1])
            preScoreListList.append(objList[2])
            preLabelsListList.append(preLabelsList)
            # preLabelsList.append(preLabels)
            # Accumulate
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            sumMsr += msr / testSampleNum
            sumAcc += acc / testSampleNum
        # Construct dictionary
        evaluateDataDict = {
            'imagesList': imagesList,
            'gtLabelsList': gtLabelsList,
            'gtBboxesList': gtBboxesList,
            'pointsListList': pointsListList,
            'preYXListList': preYXListList,
            'preHWListList': preHWListList,
            'preScoreListList': preScoreListList,
            'preLabelsListList': preLabelsListList}
        # Compute mAP and draw process
        items = RAMEvaluator.parseDict(evaluateDataDict)
        gtBboxes, preBboxes, gtLabels, preLabels = \
            RAMEvaluator.parseBboxesLabelsForSingleObj(items=items)
        mAP = RAMEvaluator.evaluate_mAP(
            gtBboxes, preBboxes, gtLabels, preLabels)
        if saveDir is not None:
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
            RAMEvaluator.drawMultiProcess(
                items=items,
                indexList=range(0, min(saveMaxNum, len(items))),
                isSeq=isSaveSeq,
                saveDir=saveDir,
                ext=ext)
            # filename = 'evaluate_data_dict' + '.pkl'
            # dataSavePath = os.path.join(saveDir,
            #                             filename)
            # with open(dataSavePath, 'wb') as output:
            #     pickle.dump(items, output)
        # Display info
        t_str = 'evaluating overlap=%f ' % (sumOverlap) + \
                'msr=%f ' % (sumMsr) + \
                'acc=%f ' % (sumAcc) + \
                'mAP=%f ' % (mAP)
        self.printLog(t_str)


def main():
    print sys.argv

    # Load dataset
    dataHomeDir = '/home/jielyu/Database/MnistScaledObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/MnistObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/Mnist-dataset'
    mnistConfig = MnistConfig(batchSize=64,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    # with tf.device('/gpu:0'):
    isTrain = True
    # Create Config object
    config = Config(isTrain=isTrain,
                    inputShape=[56, 56, 1],
                    nScale=3,
                    scaleFactor=1.5,
                    isAddContext=False,
                    minScaleSize=8,
                    targetSize=16,
                    startLearningRate=1e-3,
                    minLearningRate=1e-4,
                    monteCarloSample=10,
                    maxTrainEpoch=60,
                    keepProb=0.9)
    # Create ram object
    ram = ClassificationDetectionRAM(config=config)
    ram.setDataset(dataset)
    if isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack/CDRAM/')
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)

if __name__ == '__main__':
    main()
