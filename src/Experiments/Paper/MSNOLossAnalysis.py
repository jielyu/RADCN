# encoding: utf8

# Import system libraries
import os
import sys
import tensorflow as tf

from src.AttentionModel.ClassificationDetectionRAM \
    import ClassificationDetectionRAM as CDRAM
from src.AttentionModel.ClassificationDetectionRAM import Config
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset


IsTheLastConstraint = True


class MSOCDRAM(CDRAM):

    def __init__(self, config=Config()):
        super(MSOCDRAM, self).__init__(config=config)

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
            if IsTheLastConstraint:
                loss = -logll + ls + baselines_mse - tf.log(overlap) + ce + ads
            else:
                loss = -logll + ls + baselines_mse - tf.log(overlap) + ce
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
    dataHomeDir = '/home/jielyu/Database/MnistScaledNoisedObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/MnistObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/Mnist-dataset'
    mnistConfig = MnistConfig(batchSize=64,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    # with tf.device('/gpu:0'):
    isTrain = False

    # Parameters
    isSampling = True
    isTheLastConstraint = IsTheLastConstraint

    # The name of exp
    paraDict = {'isSample': isSampling,
                'isTheLastConstraint': isTheLastConstraint}
    expName = 'exp_msno_loss'
    for key, value in paraDict.items():
        t_str = '_' + key + '=' + str(value)
        expName += t_str
    # expName += '_supply_0'
    # Parameters
    initRange = 0.3
    nScale = 3
    scaleFactor = 2.0
    featDim = 256
    glimpStep = 10
    # Create Config object
    config = Config(isTrain=isTrain,
                    inputShape=[56, 56, 1],
                    isRandomInitial=True,
                    initialMinValue=-initRange,
                    initialMaxValue=initRange,
                    isSamplePoint=isSampling,
                    featureDim=featDim,
                    nScale=nScale,
                    scaleFactor=scaleFactor,
                    isAddContext=False,
                    minScaleSize=8,
                    maxTimeStep=glimpStep,
                    targetSize=16,
                    startLearningRate=1e-3,
                    minLearningRate=1e-4,
                    monteCarloSample=5,
                    maxTrainEpoch=120,
                    keepProb=0.9,
                    modelRelativeDir=os.path.join('output/Exp/RAM', expName))
    # Create ram object
    ram = MSOCDRAM(config=config)
    ram.setDataset(dataset)
    if isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/Exp/saveTrack',
                                expName+'_3')
        # ram.testModel(isSaveTrack=True,
        #               saveDir=trackDir)
        # ram.evaluate()
        ram.evaluate(saveDir=trackDir, saveMaxNum=300, isSaveSeq=True, ext='png')


if __name__ == '__main__':
    main()
