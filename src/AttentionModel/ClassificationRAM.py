# encoding: utf8

# System libraries
import os
import sys

import numpy as np
import tensorflow as tf

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
from Reward.ClassificationAction import ClassificationAction as ClaAction
from Reward.ClassificationAction import Config as ClaConfig
from Reward.RewardPredictionAction import Config as RWDConfig
from Reward.RewardPredictionAction \
    import RewardPredictionAction as RWDPredictor
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset
from src.SuperClass.SuperRAMConfig import SuperRAMConfig
from src.SuperClass.SuperRAModel import SuperRAModel as SuperRAM
from src.tools.Utils import Utils


# Define and implement a class
# to configure some parameters of class ObjectDetectionRAM
class Config(SuperRAMConfig):

    def __init__(self,
                 isRandomInitial=True,
                 initialMinValue=-0.3,
                 initialMaxValue=0.3,
                 inputShape=[28, 28, 1],
                 isAddContext=True,
                 nScale=3,
                 scaleFactor=1.5,
                 minScaleSize=64,
                 targetSize=64,
                 featureDim=256,
                 focusedCoordinateNum=2,
                 numCategory=10,
                 batchSize=64,
                 maxTimeStep=10,
                 lstmLayers=3,
                 isTrain=False,
                 keepProb=1.0,
                 monteCarloSample=1,
                 isSamplePoint=True,
                 samplingStd=0.2,
                 maxGradNorm=500,
                 startLearningRate=3e-3,
                 minLearningRate=1e-4,
                 learningStepFactor=0.97,
                 maxTrainEpoch=500,
                 modelFileName='ClassificationRAM.ckpt',
                 homeDir='/home/jielyu/Workspace/Python/AttentionModel',
                 modelRelativeDir='output/RAM/ClaRAM',
                 initialModelDir='',
                 logFileName='ClaTask.log'):

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

        # The shape of input images
        self.inputShape = inputShape

        # The dimension of an object
        self.numCategory = numCategory

        # The number of lstm layers
        self.lstmLayers = lstmLayers

        # Set the name of model file
        self.initialModelDir = initialModelDir

    def saveTo(self, fid):
        if not isinstance(fid, file):
            raise TypeError('Not file object')
        super(Config, self).saveTo(fid)

        fid.write('\r\n')
        fid.write('inputShape = %s\r\n' % (self.inputShape))
        fid.write('numCategory = %s\r\n' % (self.numCategory))
        fid.write('lstmLayers = %s\r\n' % (self.lstmLayers))
        fid.write('initialModelDir = %s\r\n' % (self.initialModelDir))
        fid.write('\r\n\r\n')


class ClassificationRAM(SuperRAM):

    def __init__(self, config=Config()):
        super(ClassificationRAM, self).__init__(config=config)

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
                            featureDim=self.config.featureDim)
        self.alFusion = ALFusion(config=alConfig)

        # Create lstmExtractor object
        numHiddenUnits = self.config.featureDim
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

        # Create reward predictor
        rwdConf= RWDConfig(isTrain=self.config.isTrain,
                           featureDim=numHiddenUnits,
                           keepProb=self.config.keepProb,
                           isDropout=False)
        self.rwdPredictor = RWDPredictor(config=rwdConf)

        # Create classifier
        claConf = ClaConfig(isTrain=self.config.isTrain,
                            featureDim=self.config.featureDim,
                            numCategory=self.config.numCategory,
                            keepProb=self.config.keepProb,
                            isDropout=False)
        self.claAction = ClaAction(config=claConf)

        self.images = 0  # the interface: images
        self.labels = 0  # the interface: labels

    def buildCoreGraph(self, images, points, reuse=False):

        with tf.variable_scope('ClassificationTimeStep', reuse=reuse):
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

            # Predict reward
            preRwd = self.rwdPredictor.buildGraph(lstmFeat, reuse=reuse)

            # Classify
            preProb = self.claAction.buildGraph(lstmFeat, reuse=reuse)

        # Return
        return focusedPoints, preRwd, preProb

    def buildGraph(self, name='ObjectDetectionRAM'):

        # Define the input interface
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=[None,
                                            self.config.inputShape[0],
                                            self.config.inputShape[1],
                                            self.config.inputShape[2]])

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

            preRwdList = []
            preProbList = []
            pointList = []
            pointMeanList = []
            pointList.append(points)
            pointMeanList.append(points)
            # Create RNN
            for i in range(0, self.config.maxTimeStep):
                # Loop at each time step
                if i == 0:
                    r_points, preRwd, preProb = \
                        self.buildCoreGraph(
                            images=images,
                            points=points)
                else:
                    # Parameters sharing
                    r_points, preRwd, preProb = \
                        self.buildCoreGraph(
                            images=images,
                            points=points,
                            reuse=True)
                # Obtain absolute coordinate and set range
                points_abs = r_points
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

                # Push into stake
                preRwdList.append(preRwd)
                preProbList.append(preProb)
                pointList.append(points)
                pointMeanList.append(points_mean)

            # Get the object list of a sequence
            self.preRwdList = preRwdList
            self.preProbList = preProbList
            self.pointList = pointList
            self.pointMeanList = pointMeanList
            self.preLabels = tf.argmax(self.preProbList[-1], 1)

    def buildLossGraph(self, name='LossFunction'):
        preRwdList = self.preRwdList
        preClaList = self.preProbList
        pointList = self.pointList
        pointMeanList = self.pointMeanList
        gtCla = self.labels
        loss = 0
        with tf.variable_scope(name):
            if len(preClaList) != self.config.maxTimeStep:
                raise ValueError('Not RNN output')

            # Get predicted baselines
            baselines = tf.pack(preRwdList)  # [timestep, batchsize]
            baselines = tf.transpose(baselines)  # [batchsize, timestep]

            # Compute cross-entropy
            ceList, accList, rwdList = \
                self.buildEntropyGraph(preClaList=preClaList,
                                       gtCla=gtCla)
            ce = ceList[-1]
            acc = accList[-1]
            rwd = rwdList[-1]

            # Reward
            reward = rwd
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

            # Loss function
            loss = -logll + ce + baselines_mse

        # The overlap of the last step
        reward = tf.reduce_mean(reward)
        self.loss = loss
        self.reward = reward
        self.baseline_mse = baselines_mse
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
        accTrainList = []
        lossTestList = []
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
            labels = np.tile(labels, [self.config.monteCarloSample, 1])
            feed_dict = {self.images: images, self.labels:labels}
            # Train
            sess.run(self.trainOp, feed_dict=feed_dict)

            # Test
            if i == 0 \
                    or i % self.config.minDispStep == \
                                    self.config.minDispStep-1 \
                    or self.config.isDispAlways:
                # Test on trainset
                loss, reward, bmse, logll, ce, acc, lr = \
                    sess.run((self.loss,
                              self.reward, self.baseline_mse,
                              self.logll, self.ce, self.acc,
                              self.learningRate),
                             feed_dict=feed_dict)
                # Display information
                t_str = 'step=%d ' % (i) + \
                        'epoch=%d/%d ' % (epochCnt,
                                          self.config.maxTrainEpoch) + \
                        'loss=%f ' % (loss) + \
                        'rwd=%f ' % (reward) + \
                        'bmse=%f ' % (bmse) + \
                        'logll=%f ' % (logll) + \
                        'ce=%f ' %(ce) + \
                        'acc=%f '%(acc) + \
                        'lr=%f ' % (lr)
                lossTrainList.append(loss)
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
                    teLoss, teAcc = self.testModel(sess=sess)
                    lossTestList.append(teLoss)
                    accTestList.append(teAcc)
        self.printLog('Train completely')

        # Draw Training curve
        self.saveTrainingCurve(scalarList=lossTrainList, name='train_loss')
        self.saveTrainingCurve(scalarList=accTrainList, name='train_acc')
        self.saveTrainingCurve(scalarList=lossTestList, name='test_loss')
        self.saveTrainingCurve(scalarList=accTestList, name='test_acc')

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
                         self.labels: t_labels}
            # Obtain accuracy
            loss, reward, bmse, logll, ce, acc = \
                sess.run((self.loss,
                          self.reward, self.baseline_mse,
                          self.logll, self.ce, self.acc),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumAcc += acc / testSampleNum
            # Save Track
            if isSaveTrack:
                for k in range(0, 5):
                    # Obtain accuracy
                    pointList, preLabels = \
                        sess.run((self.pointList, self.preLabels),
                                 feed_dict=feed_dict)
                    # Default direction
                    if saveDir is None:
                        saveRelativeDir = 'output/ClassificationRAM/test'
                        saveDir = os.path.join(self.config.homeDir,
                                               saveRelativeDir)
                    t_dir = os.path.join(saveDir,
                                         'batch_' + str(j) + '_' + str(k))
                    gtLabels = np.argmax(t_labels, axis=1)
                    self.saveRecurrentTrack(saveDir=t_dir,
                                            images=t_images,
                                            pointsList=pointList,
                                            preClaName=preLabels,
                                            gtBbox=t_bbox,
                                            gtClaName=gtLabels,
                                            isSaveData=isSaveData)
        t_str = 'testing ' \
                'acc=%f ' % (sumAcc)
        self.printLog(t_str)
        # Return
        return sumLoss, sumAcc


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
                    startLearningRate=1e-4,
                    minLearningRate=1e-6,
                    monteCarloSample=5,
                    maxTrainEpoch=80,
                    keepProb=0.9)
    # Create ram object
    ram = ClassificationRAM(config=config)
    ram.setDataset(dataset)
    if config.isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(config.homeDir,
                                'output/saveTrack/ClaRAM/')
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)


if __name__ == '__mailn__':
    main()
