# encoding: utf8

# Import system libraries
import os
import sys
import tensorflow as tf
# Import self-define libraries
from src.AttentionModel.Representation.CNNRepresentation \
    import CNNRepresentation as CNNExtractor
from src.AttentionModel.Representation.CNNRepresentation \
    import Config as CNNConfig
from src.AttentionModel.Representation.AppearanceLocationFusion \
    import AppearanceLocationFusion as ALFusion
from src.AttentionModel.Representation.AppearanceLocationFusion \
    import Config as ALConfig
from src.AttentionModel.Representation.LSTMRepresentation \
    import Config as LSTMConfig
from src.AttentionModel.Representation.LSTMRepresentation \
    import LSTMRepresentation as LSTMExtractor
from src.AttentionModel.FocusedPoint.FocusedPointPrediction \
    import Config as FpConfig
from src.AttentionModel.FocusedPoint.FocusedPointPrediction \
    import FocusedPointPrediction as FpPredictor
from src.AttentionModel.Reward.ClassificationAction \
    import ClassificationAction as ClaAction
from src.AttentionModel.Reward.ClassificationAction \
    import Config as ClaConfig
from src.AttentionModel.ClassificationDetectionRAM \
    import ClassificationDetectionRAM as CDRAM
from src.AttentionModel.ClassificationDetectionRAM import Config
from src.AttentionModel.Reward.ObjectDetectionReward \
    import Config as RewordConfig
from src.AttentionModel.Reward.ObjectDetectionReward \
    import ObjectDetectionReward

from src.Dataset.VOCDataset import Config as VOCConfig
from src.Dataset.VOCDataset import VOCDataset


class VOCCDRAM(CDRAM):

    def __init__(self, config=Config()):
        super(VOCCDRAM, self).__init__(config=config)

        w1_chNum = 64
        w2_chNum = 128
        w3_chNum = 4
        # Create cnnExtractor object
        filterSize = 7
        cnnConf = CNNConfig(
            inputShape=[self.eyeLikeCapture.config.targetSize,
                        self.eyeLikeCapture.config.targetSize,
                        self.config.inputShape[2]],
            w1Shape=[filterSize, filterSize, config.inputShape[2], w1_chNum],
            b1Shape=[w1_chNum],
            pool1Shape=[1, 1, 1, 1],
            w2Shape=[filterSize, filterSize, w1_chNum, w2_chNum],
            b2Shape=[w2_chNum],
            pool2Shape=[1, 1, 1, 1],
            w3Shape=[filterSize, filterSize, w2_chNum, w3_chNum],
            b3Shape=[w3_chNum],
            # pool3Shape=[1, 1, 1, 1],
            numHiddenFc1=self.config.featureDim,
            isTrain=self.config.isTrain,
            keepProb=self.config.keepProb,
            isDropout=True)
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
        claConf = ClaConfig(featureDim=config.featureDim,
                            numCategory=config.numCategory,
                            isTrain=config.isTrain,
                            keepProb=config.keepProb,
                            isDropout=True)
        self.classifier = ClaAction(config=claConf)

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

            # Compute cross-entropy
            ceList, accList, rwdList = \
                self.buildEntropyGraph(preClaList=preClaList,
                                       gtCla=gtCla)
            ce = ceList[-1]
            acc = accList[-1]
            rwd = rwdList[-1]

            # Reward
            # reward = overlap
            reward = (rwd + overlap)/2.0
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
            ol_th = 0.7
            sign_ol = tf.nn.relu(tf.sign(overlap - ol_th))
            meanSignOL = tf.reduce_mean(sign_ol) + 1e-6

            overlap = tf.reduce_mean(overlap) + 1e-6
            msr = tf.reduce_mean(msr)
            loss = -logll + ls + baselines_mse - tf.log(overlap) + ce + ads - tf.log(meanSignOL)
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

def main():
    print sys.argv
    # Load dataset
    dataHomeDir = '/home/share/Dataset/VOC-dataset/VOC2012'
    config = VOCConfig(
        enableMemSave=True,
        batchSize=64,
        datasetDir=dataHomeDir,
        isUseAllData=True,
        isTheLargestObj=True,
        maxSampleNum=100000,
        testingSampleRatio=0.3)
    dataset = VOCDataset(config=config)
    dataset.readDataset()

    isTrain = True
    expName = 'VOC_CD_RAM_Enlarge'
    # Create Config object
    config = Config(isTrain=isTrain,
                    inputShape=[500, 500, 3],
                    featureDim=512,
                    numCategory=20,
                    nScale=3,
                    scaleFactor=3.5,
                    isAddContext=False,
                    minScaleSize=32,
                    targetSize=32,
                    startLearningRate=1e-4,
                    minLearningRate=1e-4,
                    monteCarloSample=1,
                    maxTrainEpoch=400,
                    keepProb=0.5,
                    modelRelativeDir=os.path.join('output/RAM', expName))
    # Create ram object
    ram = VOCCDRAM(config=config)
    ram.setDataset(dataset)
    if isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack',
                                expName)
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)


if __name__ == '__main__':
    main()
