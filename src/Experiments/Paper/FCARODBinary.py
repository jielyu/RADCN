# encoding: utf8

# Import system libraries
import os
import sys

import tensorflow as tf
# Import self-define libraries
from src.AttentionModel.FocusedPoint.FocusedPointPrediction \
    import Config as FpConfig
from src.AttentionModel.FocusedPoint.FocusedPointPrediction \
    import FocusedPointPrediction as FpPredictor
from src.AttentionModel.ObjectDetectionRAM import Config
from src.AttentionModel.ObjectDetectionRAM import ObjectDetectionRAM as ODRAM
from src.AttentionModel.Representation.AppearanceLocationFusion \
    import AppearanceLocationFusion as ALFusion
from src.AttentionModel.Representation.AppearanceLocationFusion \
    import Config as ALConfig
from src.AttentionModel.Representation.CNNRepresentation \
    import CNNRepresentation as CNNExtractor
from src.AttentionModel.Representation.CNNRepresentation \
    import Config as CNNConfig
from src.AttentionModel.Representation.LSTMRepresentation \
    import Config as LSTMConfig
from src.AttentionModel.Representation.LSTMRepresentation \
    import LSTMRepresentation as LSTMExtractor
from src.AttentionModel.Reward.ObjectDetectionReward \
    import Config as RewordConfig
from src.AttentionModel.Reward.ObjectDetectionReward \
    import ObjectDetectionReward
from src.Dataset.FCARDataset import Config as FCARConfig
from src.Dataset.FCARDataset import FCARDataset


class FCARDetectionRAM(ODRAM):

    def __init__(self, config=Config()):
        super(FCARDetectionRAM, self).__init__(config=config)

        w1_chNum = 64
        w2_chNum = 128
        w3_chNum = 4
        # Create cnnExtractor object
        cnnConf = CNNConfig(
            inputShape=[self.eyeLikeCapture.config.targetSize,
                        self.eyeLikeCapture.config.targetSize,
                        self.config.inputShape[2]],
            w1Shape=[5, 5, self.config.inputShape[2], w1_chNum],
            b1Shape=[w1_chNum],
            pool1Shape=[1, 1, 1, 1],
            w2Shape=[5, 5, w1_chNum, w2_chNum],
            b2Shape=[w2_chNum],
            pool2Shape=[1, 1, 1, 1],
            w3Shape=[5, 5, w2_chNum, w3_chNum],
            b3Shape=[w3_chNum],
            # pool3Shape=[1, 1, 1, 1],
            numHiddenFc1=self.config.featureDim,
            isTrain=self.config.isTrain,
            keepProb=self.config.keepProb,
            isDropout=False)
        self.cnnExtractor = []
        for i in range(0, self.config.nScale):
            self.cnnExtractor.append(CNNExtractor(config=cnnConf))

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
        # numHiddenUnits = self.config.featureDim+\
        #                  self.config.focusedCoordinateNum

        lstmConf = LSTMConfig(numHiddenUnits=numHiddenUnits,
                              visualFeatDim=self.config.featureDim,
                              isTrain=self.config.isTrain,
                              fcKeepProb=self.config.keepProb,
                              lstmKeepProb=self.config.keepProb,
                              isDropout=True)
        self.lstmExtractor = LSTMExtractor(config=lstmConf)

        # Create focused-point prediction object
        fpConf = FpConfig(featureDim=numHiddenUnits,
                          numHiddenUnits=512,
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
            ol_th = 0.7
            sign_ol = tf.nn.relu(tf.sign(overlap-ol_th))
            meanSignOL = tf.reduce_mean(sign_ol) + 1e-6

            overlap = tf.reduce_mean(overlap) + 1e-6
            msr = tf.reduce_mean(msr)
            loss = -logll + ls + baselines_mse - tf.log(overlap) + ads - tf.log(meanSignOL)
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


def main():
    print sys.argv

    # Load dataset
    dataHomeDir = '/home/jielyu/Database/FCAR-dataset'
    fcarConfig = FCARConfig(
        enableMemSave=True,
        batchSize=64,
        datasetDir=dataHomeDir,
        maxSampleNum=30000)
    dataset = FCARDataset(config=fcarConfig)
    dataset.readDataset()

    isTrain = False
    # Create Config
    step = 10
    # expName = 'exp_fcar_binary_step='+str(step)
    expName = 'exp_fcar_binary'
    ramConfig = Config(isTrain=isTrain,
                       batchSize=64,
                       isAbsoluteAttention=True,
                       inputShape=[800, 800, 3],
                       nScale=3,
                       featureDim=512,
                       scaleFactor=4.5,
                       isAddContext=False,
                       minScaleSize=32,
                       targetSize=32,
                       maxTimeStep=step,
                       startLearningRate=1e-4,
                       minLearningRate=1e-4,
                       learningStepFactor=0.99,
                       monteCarloSample=1,
                       maxTrainEpoch=300,
                       keepProb=0.8,
                       modelRelativeDir=os.path.join('output/Exp/RAM',
                                                     expName))
    # ramConfig.initialModelDir = ramConfig.modelDir

    # Create ram object
    ram = FCARDetectionRAM(config=ramConfig)
    ram.setDataset(dataset)

    if ramConfig.isTrain:
        # Train model
        ram.trainModel()
    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/Exp/saveTrack',
                                expName+'_3')
        # ram.testModel(isSaveTrack=True,
        #               saveDir=trackDir)
        ram.evaluate(saveDir=trackDir, saveMaxNum=300, isSaveSeq=True, ext='png')

if __name__ == '__main__':
    main()
