# encoding: utf8

# Import system libraries
import os
import sys
import pickle
import numpy as np
import tensorflow as tf

# Self-define libraries
from src.SuperClass.SuperRAMConfig import SuperRAMConfig
from src.SuperClass.SuperRAModel import SuperRAModel
from InputChannel.EyeLikeCapture import Config as EyeConfig
from InputChannel.EyeLikeCapture import EyeLikeCapture as Eye
from Representation.AppearanceLocationFusion \
    import AppearanceLocationFusion as ALFusion
from Representation.AppearanceLocationFusion \
    import Config as ALConfig
# from Representation.CNNRepresentation import CNNRepresentation as CNNExtractor
# from Representation.CNNRepresentation import Config as CNNConfig
from Representation.RWCNNRepresentation import CNNRepresentation as CNNExtractor
from Representation.RWCNNRepresentation import Config as CNNConfig
from Representation.LSTMRepresentation import Config as LSTMConfig
from Representation.LSTMRepresentation \
    import LSTMRepresentation as LSTMExtractor
from FocusedPoint.FocusedPointPrediction import Config as FpConfig
from FocusedPoint.FocusedPointPrediction \
    import FocusedPointPrediction as FpPredictor
from Reward.ObjectDetectionReward import Config as RewordConfig
from Reward.ObjectDetectionReward import ObjectDetectionReward
from src.AttentionModel.Reward.ClassificationAction import ClassificationAction as ClaAction
from src.AttentionModel.Reward.ClassificationAction import Config as ClaConfig
from src.tools.Utils import Utils
from src.AttentionModel.RAMEvaluator import RAMEvaluator

from src.Dataset.CUB200Dataset import Config as CUB200Config
from src.Dataset.CUB200Dataset import CUB200Dataset


class Config(SuperRAMConfig):

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
                 objectDim=5,
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
                 logFileName='CDTask.log'
                 ):
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
            logFileName=logFileName
        )
        self.isAbsoluteAttention = False
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
        self.numCategory = numCategory

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
        fid.write('numCategory = %s\r\n' % (self.numCategory))
        fid.write('\r\n\r\n')


class RWCDRAM(SuperRAModel):

    def __init__(self, config=Config()):
        super(RWCDRAM, self).__init__(config=config)

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
            # w1Shape=[5, 5, self.config.inputShape[2], 32],
            # b1Shape=[32],
            # pool1Shape=[1, 1, 1, 1],
            # w2Shape=[5, 5, 32, 64],
            # b2Shape=[64],
            # pool2Shape=[1, 1, 1, 1],
            # w3Shape=[5, 5, 64, 4],
            # b3Shape=[4],
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
        # Classification action
        claConf = ClaConfig(featureDim=config.featureDim,
                            numCategory=config.numCategory,
                            isTrain=config.isTrain,
                            keepProb=config.keepProb,
                            isDropout=True)
        self.classifier = ClaAction(config=claConf)

        self.labels = 0

        # self.trainingSampleNum = 1000   # the umber of training samples
        self.images = 0                 # the interface: images
        self.objects = 0                # the interface: bbox
        self.objList = 0                # the results of all steps
        self.overlap = 0                # the mean overlap of the last step
        self.msr = 0                    # the mean square of position error
        self.ls = 0                     # least square

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
                # Add summary for images
                tf.summary.image(name='ImgScale'+str(i),
                                 tensor=multiScaleImages[i],
                                 max_outputs=1)
                name = 'ConvScale_' + str(i)
                cnnFeat.append(
                self.cnnExtractor[i].buildGraph(
                    multiScaleImages[i], name=name, reuse=reuse))

            # Concat multi-scale CNN representations
            concatedCNNFeat = tf.concat(axis=1, values=cnnFeat)
            tf.summary.histogram(name='CNNFeat', values=concatedCNNFeat)

            # Fusion
            fusionFeat = self.alFusion.buildGraph(concatedCNNFeat, points, name='FuseNet', reuse=reuse)
            tf.summary.histogram(name='FusedFeat', values=fusionFeat)

            # Extract LSTM representation
            lstmFeat = self.lstmExtractor.buildGraph(fusionFeat, name='LSTMNet', reuse=reuse)
            tf.summary.histogram(name='LSTMFeat', values=lstmFeat)

            # Predict the next focused points
            focusedPoints = self.fpPredictor.buildGraph(lstmFeat, name='FPNet', reuse=reuse)

            # Generate boxes and scores of objects
            objs = self.objectDetectionReward.buildGraph(lstmFeat, name='ODNet', reuse=reuse)

            claProb = self.classifier.buildGraph(lstmFeat, name='ClasNet', reuse=reuse)
            tf.summary.histogram(name='preClasProb', values=claProb)
            tf.summary.histogram(name='preBbox', values=tf.concat(values=objs, axis=-1))

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
        tf.summary.histogram(name='GtProb', values=self.labels)
        tf.summary.histogram(name='GtBbox', values=self.objects)
        tf.summary.histogram(name='InputImage', values=self.images)

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
            tf.summary.histogram(name='preLabel', values=tf.argmax(claList[-1], axis=-1))
            tf.summary.histogram(name='gtLabel', values=tf.argmax(tf.squeeze(self.labels),
                                                                  axis=-1))

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
        acc = 0
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
            tf.summary.scalar(name='ls', tensor=ls)

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
            tf.summary.scalar(name='ce', tensor=ce)
            acc = accList[-1]
            tf.summary.scalar(name='acc', tensor=acc)
            rwd = rwdList[-1]

            # Reward
            reward = overlap
            tf.summary.scalar(name='reward', tensor=tf.reduce_mean(reward))
            # reward = (rwd + overlap)/2.0
            # reward = rwd
            rewards = tf.expand_dims(reward, 1)  # [batchsize, timestep]
            rewards = tf.tile(rewards,
                              (1, self.config.maxTimeStep))

            bias = rewards - tf.stop_gradient(baselines)
            baselines_mse = tf.reduce_mean(
                tf.square(rewards - baselines)
            )
            tf.summary.scalar(name='baseline_mse', tensor=baselines_mse)

            # Compute log likelihood of points
            logll = self.buildLogLikelihoodGraph(pointList=pointList,
                                                 pointMeanList=pointMeanList)
            logll = tf.reduce_mean(logll * bias)
            tf.summary.scalar(name='logll', tensor=logll)
            # logll = tf.reduce_mean(logll * rewards)
            # logll = tf.reduce_mean(logll*tf.square(rewards - baselines))

            # Compute attention distance
            ads = self.buildAttentionDistanceGraph(pointMeanList, gtObjs)
            ads = tf.reduce_mean(ads[-1])
            tf.summary.scalar(name='ads', tensor=ads)

            # Loss function
            overlap = tf.reduce_mean(overlap) + 1e-6
            tf.summary.scalar(name='overlap', tensor=overlap)
            msr = tf.reduce_mean(msr)
            tf.summary.scalar(name='msr', tensor=msr)
            loss = -logll + ls + baselines_mse - tf.log(overlap) + ce + ads
            # loss = - tf.log(overlap) + ce + ads
            # loss = -logll + ce + baselines_mse
            # loss = -logll + ls + baselines_mse - tf.log(overlap)

            # Regular loss
            reg_loss = 0.001*tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
            tf.summary.scalar(name='reg_loss', tensor=reg_loss)
            loss += reg_loss
            tf.summary.scalar(name='loss', tensor=loss)

        # The overlap of the last step
        self.loss = loss
        self.overlap = overlap
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
            _, summary, \
            loss, overlap, acc, lr, \
            claList, gtlabels \
                = sess.run((self.trainOp, self.merge,
                            self.loss, self.overlap, self.acc, self.learningRate,
                            self.claList, self.labels
                                   ), feed_dict=feed_dict)
            self.summary.add_summary(summary=summary, global_step=i)

            # Test on trainset
            # loss, overlap, msr, reward, bmse, ls, \
            # ads, logll, ce, acc, lr, claList, gtlabels = \
            #     sess.run((),
            #              feed_dict=feed_dict)
            # print claList[-1], np.argmax(claList[-1], axis=-1), gtlabels, np.argmax(np.squeeze(gtlabels), axis=-1)
            # Display information
            t_str = 'step=%d ' % (i) + \
                    'epoch=%d/%d ' % (epochCnt,
                                      self.config.maxTrainEpoch) + \
                    'loss=%f ' % (loss) + \
                    'ol=%f ' % (overlap) + \
                    'acc=%f ' % (acc) + \
                    'lr=%f ' % (lr)
            lossTrainList.append(loss)
            overlapTrainList.append(overlap)
            accTrainList.append(acc)
            self.printLog(t_str)

            # Test
            mds = self.config.minDispStep
            if i % mds == mds-1:
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
            loss, overlap, acc = \
                sess.run((self.loss, self.overlap, self.acc),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            # sumMsr += msr / testSampleNum
            sumAcc += acc / testSampleNum

            if isSaveTrack:
                for k in range(0, 3):
                    # Obtain accuracy
                    t_overlap, pointList, objList, preLabels = \
                        sess.run((self.overlap,
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
            loss, overlap, \
            acc, pointList, \
            objList, preLabelsList, preLabels = \
                sess.run((self.loss, self.overlap,
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
            # sumMsr += msr / testSampleNum
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
                'acc=%f ' % (sumAcc) + \
                'mAP=%f ' % (mAP)
        self.printLog(t_str)


def main():
    print sys.argv
    # Load dataset
    dataHomeDir = '/media/home_bak/jielyu/Database/CUB200-dataset'
    batchSize = 64
    imgSize = [448, 448, 3]
    cub200Config = CUB200Config(
        dataHomeDir=dataHomeDir,
        batchSize=batchSize,
        imgSize=imgSize,
        subsetName='CUB_200_2011')
    dataset = CUB200Dataset(config=cub200Config)
    dataset.readDataset()

    # with tf.device('/gpu:0'):
    isTrain = True

    # Parameters
    initRange = 0.3
    nScale = 3
    scaleFactor = 2.5
    featDim = 1024
    glimpStep = 10
    # The name of exp
    paraDict = {'initRange': initRange,
                'nScale': nScale,
                'scaleFactor': scaleFactor,
                'featDim': featDim,
                'glimpStep': glimpStep}
    expName = 'RWCDRAM_reg_l1_nosam'
    # for key, value in paraDict.items():
    #     t_str = '_' + key + '=' + str(value)
    #     expName += t_str
    # expName += '_supply_0'
    # Create Config object
    config = Config(isTrain=isTrain,
                    batchSize=batchSize,
                    numCategory=200,
                    inputShape=imgSize,
                    isSamplePoint=False,
                    isRandomInitial=False,
                    initialMinValue=-initRange,
                    initialMaxValue=initRange,
                    featureDim=featDim,
                    nScale=nScale,
                    scaleFactor=scaleFactor,
                    isAddContext=False,
                    minScaleSize=48,
                    maxTimeStep=glimpStep,
                    targetSize=48,
                    startLearningRate=1e-5,
                    minLearningRate=1e-5,
                    maxGradNorm=50000,
                    monteCarloSample=1,
                    maxTrainEpoch=200,
                    keepProb=0.8,
                    modelRelativeDir=os.path.join('output/RAM', expName))
    # config.initialModelDir = config.modelDir
    # Create ram object
    ram = RWCDRAM(config=config)
    ram.setDataset(dataset)
    if isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack',
                                expName)
        # ram.testModel(isSaveTrack=True,
        #               saveDir=trackDir)

        # ram.testModel()
        # Evaluate
        dataSaveDir = trackDir
        # ram.evaluate()
        ram.evaluate(saveDir=dataSaveDir, saveMaxNum=100, isSaveSeq=False)


if __name__ == '__main__':
    main()
