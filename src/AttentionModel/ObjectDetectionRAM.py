# encoding:utf-8

# System libraries
import os
import sys

# 3rd-part libraries
import pickle
import numpy as np
import tensorflow as tf

# Self-define libraries
from FocusedPoint.FocusedPointPrediction import Config as FpConfig
from FocusedPoint.FocusedPointPrediction \
    import FocusedPointPrediction as FpPredictor
from InputChannel.EyeLikeCapture import Config as EyeConfig
from InputChannel.EyeLikeCapture import EyeLikeCapture as Eye
from Representation.AppearanceLocationFusion \
    import AppearanceLocationFusion as ALFusion
from Representation.AppearanceLocationFusion \
    import Config as ALConfig
from Representation.CNNRepresentation import CNNRepresentation as CNNExtractor
from Representation.CNNRepresentation import Config as CNNConfig
from Representation.LSTMRepresentation import Config as LSTMConfig
from Representation.LSTMRepresentation \
    import LSTMRepresentation as LSTMExtractor
from Reward.ObjectDetectionReward import Config as RewordConfig
from Reward.ObjectDetectionReward import ObjectDetectionReward
from src.Dataset.FCARDataset import Config as FCARConfig
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.FCARDataset import FCARDataset
from src.Dataset.MnistObjectDataset import MnistObjectDataset
from src.SuperClass.SuperRAMConfig import SuperRAMConfig
from src.SuperClass.SuperRAModel import SuperRAModel as SuperRAM
from src.tools.Utils import Utils
from RAMEvaluator import RAMEvaluator

# Define and implement a class
# to configure some parameters of class ObjectDetectionRAM
class Config(SuperRAMConfig):

    def __init__(self,
                 isRandomInitial=True,
                 initialMinValue=-0.3,
                 initialMaxValue=0.3,
                 isAbsoluteAttention=False,
                 inputShape=[56, 56, 1],
                 isAddContext=False,
                 nScale=3,
                 scaleFactor=1.5,
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
                 startLearningRate=1e-3,
                 minLearningRate=1e-4,
                 learningStepFactor=0.97,
                 maxTrainEpoch=40,
                 modelFileName='ODRAM.ckpt',
                 homeDir='/home/jielyu/Workspace/Python/AttentionModel',
                 modelRelativeDir='output/RAM/ODRAM',
                 initialModelDir='',
                 logFileName='-ODTask.log'):
        super(Config, self).__init__(
            isRandomInitial=isRandomInitial,
            initialMinValue=initialMinValue,
            initialMaxValue=initialMaxValue,
            isAddContext=isAddContext,
            nScale=nScale,
            scaleFactor=scaleFactor,
            minScaleSize=minScaleSize,
            targetSize=targetSize,
            featureDim=featureDim,
            focusedCoordinateNum=focusedCoordinateNum,
            batchSize=batchSize,
            maxTimeStep=maxTimeStep,
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
            logFileName=logFileName)

        self.isAbsoluteAttention = isAbsoluteAttention
        # The shape of input images
        self.inputShape = inputShape
        # The dimension of an object
        self.objectDim = objectDim
        # The flag of single object
        self.maxObjectNum = maxObjectNum
        # Select the method to fuse visual and location feature
        self.fusionMethod = fusionMethod
        # The number of lstm layers
        self.lstmLayers = lstmLayers
        # Set the name of model file
        self.initialModelDir = initialModelDir

    def saveTo(self, fid):
        if not isinstance(fid, file):
            raise TypeError('Not file object')
        super(Config, self).saveTo(fid)

        fid.write('\r\n')
        fid.write('Configuration for Object detection task:\r\n')
        fid.write('isAbsoluteAttention = %s\r\n' % (self.isAbsoluteAttention))
        fid.write('inputShape = %s\r\n' % (self.inputShape))
        fid.write('objectDim = %s\r\n' % (self.objectDim))
        fid.write('maxObjectNum = %s\r\n' % (self.maxObjectNum))
        fid.write('fusionMethod = %s\r\n' % (self.fusionMethod))
        fid.write('lstmLayers = %s\r\n' % (self.lstmLayers))
        fid.write('initialModelDir = %s\r\n' % (self.initialModelDir))
        fid.write('\r\n\r\n')


# Define and implement a class to build a graph
# for object detection via RAM in tensorflow
class ObjectDetectionRAM(SuperRAM):

    def __init__(self, config=Config()):
        super(ObjectDetectionRAM, self).__init__(config=config)

        # Create eye-like object
        eyeConf = EyeConfig(isAddContext=self.config.isAddContext,
                            nScale=self.config.nScale,
                            scaleFactor=self.config.scaleFactor,
                            minScaleSize=self.config.minScaleSize,
                            targetSize=self.config.targetSize)
        self.eyeLikeCapture = Eye(config=eyeConf)

        # Create cnnExtractor object
        cnnConf = CNNConfig(
            inputShape=[self.eyeLikeCapture.config.targetSize,
                        self.eyeLikeCapture.config.targetSize,
                        self.config.inputShape[2]],
            w1Shape=[5, 5, self.config.inputShape[2], 32],
            b1Shape=[32],
            pool1Shape=[1, 1, 1, 1],
            w2Shape=[5, 5, 32, 64],
            b2Shape=[64],
            pool2Shape=[1, 1, 1, 1],
            w3Shape=[5, 5, 64, 4],
            b3Shape=[4],
            # pool3Shape=[1, 1, 1, 1],
            numHiddenFc1=self.config.featureDim,
            isTrain=self.config.isTrain,
            keepProb=self.config.keepProb,
            isDropout=False)
        self.cnnExtractor = []
        for i in range(0, self.config.nScale):
            self.cnnExtractor.append(CNNExtractor(config=cnnConf))
        # self.cnnExtractor = CNNExtractor(config=cnnConf)

        # Create fusion net
        dimCNNFeat = self.config.featureDim
        dimVisualFeat = self.config.nScale * dimCNNFeat
        focusedCoordinateNum = self.config.focusedCoordinateNum
        alConfig = ALConfig(visualFeatDim=dimVisualFeat,
                            locationFeatDim=focusedCoordinateNum,
                            featureDim=self.config.featureDim,
                            method=config.fusionMethod)
        self.alFusion = ALFusion(config=alConfig)

        # Create lstmExtractor object

        numHiddenUnits = self.config.featureDim
        # numHiddenUnits = self.config.featureDim+\
        #                  self.config.focusedCoordinateNum

        lstmConf = LSTMConfig(numHiddenUnits=numHiddenUnits,
                              visualFeatDim=self.config.featureDim,
                              isTrain=self.config.isTrain,
                              fcKeepProb=self.config.keepProb,
                              lstmKeepProb=self.config.keepProb,
                              isDropout=False)
        self.lstmExtractor = LSTMExtractor(config=lstmConf)

        # Create focused-point prediction object
        fpConf = FpConfig(featureDim=numHiddenUnits,
                          numHiddenUnits=64,
                          coordinateDim=self.config.focusedCoordinateNum,
                          isTrain=self.config.isTrain,
                          keepProb=self.config.keepProb,
                          isDropout=True)
        self.fpPredictor = FpPredictor(config=fpConf)

        # Create reward object for object detection task
        rewardConf = RewordConfig(featureDim=numHiddenUnits,
                                  numHiddenUnits=128,
                                  objectDim=self.config.objectDim,
                                  isTrain=self.config.isTrain,
                                  keepProb=self.config.keepProb,
                                  isDropout=True)
        self.objectDetectionReward = ObjectDetectionReward(config=rewardConf)

        # self.trainingSampleNum = 1000   # the umber of training samples
        self.images = 0                 # the interface: images
        self.objects = 0                # the interface: bbox
        self.objList = 0                # the results of all steps
        self.overlap = 0                # the mean overlap of the last step
        self.msr = 0                    # the mean square of position error
        self.ls = 0                     # least square

    def buildCoreGraph(self, images, points,
                       name='ClassificationTimeStep',
                       reuse=False):
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
            concatedCNNFeat = tf.concat(concat_dim=1, values=cnnFeat)

            # Fusion
            fusionFeat = self.alFusion.buildGraph(concatedCNNFeat, points, reuse=reuse)

            # Extract LSTM representation
            lstmFeat = self.lstmExtractor.buildGraph(fusionFeat, reuse=reuse)

            # Predict the next focused points
            focusedPoints = self.fpPredictor.buildGraph(lstmFeat, reuse=reuse)

            # Generate boxes and scores of objects
            objs = self.objectDetectionReward.buildGraph(lstmFeat, reuse=reuse)

            # Return
            return focusedPoints, objs

    def buildGraph(self, name='ObjectDetectionRAM'):
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
            pointList = []
            pointMeanList = []
            pointList.append(points)
            pointMeanList.append(points)
            # Create RNN
            for i in range(0, self.config.maxTimeStep):
                # Loop at each time step
                if i == 0:
                    r_points, objs = \
                        self.buildCoreGraph(
                            images=images,
                            points=points)
                else:
                    # Parameters sharing
                    r_points, objs = \
                        self.buildCoreGraph(
                            images=images,
                            points=points,
                            reuse=True)
                # Obtain absolute coordinate and set range
                if self.config.isAbsoluteAttention:
                    points_abs = r_points
                    points_mean = tf.clip_by_value(points_abs, -1.0, 1.0)
                else:
                    points_abs = points + r_points
                    points_mean = tf.clip_by_value(points_abs, -1.0, 1.0)
                    points_mean = tf.stop_gradient(points_mean)

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
                pointList.append(points)
                pointMeanList.append(points_mean)

            # Get the object list of a sequence
            self.objList = [yxList, hwList, scoreList]
            self.pointList = pointList
            self.pointMeanList = pointMeanList

    def buildLossGraph(self, name='LossFunction'):

        preYXList = self.objList[0]
        preHWList = self.objList[1]
        preScoreList = self.objList[2]
        pointList = self.pointList
        pointMeanList = self.pointMeanList
        gtObjs = self.objects
        loss = 0
        overlap = 0
        with tf.variable_scope(name):
            if len(preYXList) != self.config.maxTimeStep:
                raise ValueError('Not RNN output')

            # Get predicted baselines
            baselines = tf.pack(preScoreList)  # [timestep, batchsize]
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

            # Reward
            reward = overlap
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

            # Compute attention distance
            ads = self.buildAttentionDistanceGraph(pointMeanList, gtObjs)
            ads = tf.reduce_mean(ads[-1])

            # Loss function
            overlap = tf.reduce_mean(overlap) + 1e-6
            msr = tf.reduce_mean(msr)
            loss = -logll + ls + baselines_mse - tf.log(overlap) + ads
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
        lossTestList = []
        overlapTestList = []
        for i in range(0, self.config.maxTrainTimes):

            epochCnt = i // self.config.minDispStep
            # Obtain a batch of data
            images, bbox = \
                self.dataset.trainset.getNextBatch()
            bbox = Utils.convertToYXHW(bbox)
            images, bbox = Utils.normalizeImagesAndBbox(images, bbox)
            images = \
                np.tile(images, [self.config.monteCarloSample, 1, 1, 1])
            bbox = np.tile(bbox, [self.config.monteCarloSample, 1, 1])
            feed_dict = {self.images: images, self.objects: bbox}
            # Train
            sess.run(self.trainOp, feed_dict=feed_dict)

            # Test
            if i == 0 \
                    or i % self.config.minDispStep == \
                                    self.config.minDispStep-1 \
                    or self.config.isDispAlways:
                # Test on trainset
                loss, overlap, msr, reward, bmse, ls, ads, logll, lr = \
                    sess.run((self.loss, self.overlap, self.msr,
                              self.reward, self.baseline_mse,
                              self.ls, self.ads, self.logll,
                              self.learningRate),
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
                        'lr=%f ' % (lr)
                lossTrainList.append(loss)
                overlapTrainList.append(overlap)
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
                    teLoss, teOverlap = self.testModel(sess=sess)
                    lossTestList.append(teLoss)
                    overlapTestList.append(teOverlap)
        self.printLog('Train completely')

        # Draw Training curve
        self.saveTrainingCurve(scalarList=lossTrainList,
                               name='train_loss')
        self.saveTrainingCurve(scalarList=overlapTrainList,
                               name='train_overlap')
        self.saveTrainingCurve(scalarList=lossTestList,
                               name='test_loss')
        self.saveTrainingCurve(scalarList=overlapTestList,
                               name='test_overlap')

    def testModel(self,
                  modelDir='',
                  testset=None,
                  isSaveTrack=False,
                  isSaveData=False,
                  saveDir=None,
                  sess=None):
        print 'testing on the testset ...'
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
        testSampleNum = \
            (numTestSample // self.config.batchSize) + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox = \
                self.dataset.testset.getNextBatch()
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.objects: t_bbox}
            # Obtain accuracy
            loss, overlap, msr, reward, bmse, ls, logll = \
                sess.run((self.loss, self.overlap, self.msr,
                          self.reward, self.baseline_mse,
                          self.ls, self.logll),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            sumMsr += msr / testSampleNum

            if isSaveTrack:
                for k in range(0, 3):
                    # Obtain accuracy
                    t_overlap, t_msr, pointList, objList = \
                        sess.run((self.overlap, self.msr, self.pointList,
                                  self.objList),
                                 feed_dict=feed_dict)
                    if saveDir is None:
                        saveDir = os.path.join(self.config.homeDir,
                                               'output/CDRAM/test')
                    t_dir = os.path.join(saveDir,
                                         'batch_' + str(j) + '_' + str(k))
                    self.saveRecurrentTrack(saveDir=t_dir,
                                            images=t_images,
                                            pointsList=pointList,
                                            YXList=objList[0],
                                            HWList=objList[1],
                                            scoreList=objList[2],
                                            gtBbox=t_bbox,
                                            isSaveData=isSaveData)
        t_str = 'testing overlap=%f ' % (sumOverlap) + \
                'msr=%f ' % (sumMsr)
        self.printLog(t_str)

        return sumLoss, sumOverlap

    def evaluate(self,
                 modelDir='',
                 sess=None,
                 saveDir=None,
                 saveMaxNum=100,
                 isSaveSeq=False,
                 ext='png'):
        print 'evaluating on the testset ...'
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

        imagesList = []
        gtBboxesList = []
        pointsListList = []
        preYXListList = []
        preHWListList = []
        preScoreListList = []
        # Test on testset
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumOverlap = 0
        sumMsr = 0
        testSampleNum = \
            (numTestSample // self.config.batchSize) + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox = \
                self.dataset.testset.getNextBatch()
            imagesList.append(t_images)
            gtBboxesList.append(t_bbox)
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.objects: t_bbox}
            # Obtain accuracy
            loss, overlap, msr, reward, bmse, ls, logll, \
            pointList, objList = \
                sess.run((self.loss, self.overlap, self.msr,
                          self.reward, self.baseline_mse,
                          self.ls, self.logll,
                          self.pointList, self.objList,),
                         feed_dict=feed_dict)
            pointsListList.append(pointList)
            preYXListList.append(objList[0])
            preHWListList.append(objList[1])
            preScoreListList.append(objList[2])
            # Accumulate
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            sumMsr += msr / testSampleNum
        # Construct dictionary
        evaluateDataDict = {
            'imagesList': imagesList,
            'gtBboxesList': gtBboxesList,
            'pointsListList': pointsListList,
            'preYXListList': preYXListList,
            'preHWListList': preHWListList,
            'preScoreListList': preScoreListList}

        # filename = 'evaluate_data_dict' + '.pkl'
        # dataSavePath = os.path.join(saveDir,
        #                             filename)
        # with open(dataSavePath, 'wb') as output:
        #     pickle.dump(evaluateDataDict, output)
        #
        # dataName = 'evaluate_data_dict.pkl'
        # dataPath = os.path.join(saveDir, dataName)
        # with open(dataPath, 'r') as input:
        #     evaluateDataDict = pickle.load(input)

        # Compute mAP and draw process
        items = RAMEvaluator.parseDict(evaluateDataDict)
        gtBboxes, preBboxes, gtLabels, preLabels = \
            RAMEvaluator.parseBboxesLabelsForSingleObj(items=items)
        if saveDir is None:
            RAMEvaluator.evaluate_IoURecall(gtBboxes, preBboxes)
        else:
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
            path = os.path.join(saveDir, 'IoU_Recall.png')
            RAMEvaluator.evaluate_IoURecall(
                gtBboxes, preBboxes, savePath=path)
            # Draw process
            RAMEvaluator.drawMultiProcess(
                items=items,
                indexList=range(0, min(saveMaxNum, len(items))),
                isSeq=isSaveSeq,
                saveDir=saveDir,
                ext=ext)
        # Display info
        t_str = 'testing overlap=%f ' % (sumOverlap) + \
                'msr=%f ' % (sumMsr)
        self.printLog(t_str)


# The main function of the demo
def main():
    print sys.argv

    # Load dataset
    dataHomeDir = '/home/jielyu/Database/MnistScaledObject-dataset'
    mnistConfig = MnistConfig(batchSize=64,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    ramConfig = Config(isTrain=True,
                       isAbsoluteAttention=True,
                       inputShape=[56, 56, 1],
                       nScale=3,
                       scaleFactor=1.5,
                       isAddContext=False,
                       minScaleSize=8,
                       targetSize=16,
                       startLearningRate=1e-3,
                       minLearningRate=1e-4,
                       monteCarloSample=10,
                       maxTrainEpoch=40,
                       keepProb=0.9)

    with tf.device('/gpu:1'):
        # Create ram object
        ram = ObjectDetectionRAM(config=ramConfig)
        ram.setDataset(dataset)

        if ramConfig.isTrain:
            # Train model
            ram.trainModel()
        else:
            trackDir = os.path.join(ram.config.homeDir,
                                    'output/saveTrack/ODRAM/')
            ram.testModel(isSaveTrack=True,
                          saveDir=trackDir)


def main_1():
    print sys.argv

    # Load dataset
    dataHomeDir = '/home/jielyu/Database/FCAR-dataset'
    fcarConfig = FCARConfig(batchSize=64,
                            datasetDir=dataHomeDir,
                            maxSampleNum=30000)
    dataset = FCARDataset(config=fcarConfig)
    dataset.readDataset()

    homeDir = '/home/jielyu/Workspace/Python/AttentionModel/'
    modelRelativeDir = 'output/RAM/FCAR_ODRAM'
    ramConfig = Config(isTrain=True,
                       isAbsoluteAttention=True,
                       inputShape=[800, 800, 3],
                       nScale=3,
                       scaleFactor=2.5,
                       isAddContext=False,
                       minScaleSize=64,
                       targetSize=64,
                       startLearningRate=1e-4,
                       minLearningRate=1e-5,
                       monteCarloSample=1,
                       maxTrainEpoch=100,
                       keepProb=0.9,
                       homeDir=homeDir,
                       modelRelativeDir=modelRelativeDir)

    # with tf.device('/gpu:2'):
    # Create ram object
    ram = ObjectDetectionRAM(config=ramConfig)
    ram.setDataset(dataset)

    if ramConfig.isTrain:
        # Train model
        ram.trainModel()
    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack/FCAR_ODRAM/')
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)


# The entry of the demo
if __name__ == '__main__':
    main()
