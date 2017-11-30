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
from src.AttentionModel.Reward.ObjectDetectionReward \
    import Config as RewordConfig
from src.AttentionModel.Reward.ObjectDetectionReward \
    import ObjectDetectionReward
from src.AttentionModel.Reward.ClassificationAction \
    import ClassificationAction as ClaAction
from src.AttentionModel.Reward.ClassificationAction \
    import Config as ClaConfig
from src.AttentionModel.ClassificationDetectionRAM \
    import ClassificationDetectionRAM as CDRAM
from src.AttentionModel.ClassificationDetectionRAM import Config
from src.Dataset.VOCDataset import Config as VOCConfig
from src.Dataset.VOCDataset import VOCDataset


class VOCCDRAM(CDRAM):

    def __init__(self, config=Config()):
        super(VOCCDRAM, self).__init__(config=config)
        #
        w1_chNum = 64
        w2_chNum = 128
        w3_chNum = 4
        # Create cnnExtractor object
        filterSize = 5
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
        # alConfig = ALConfig(visualFeatDim=dimVisualFeat,
        #                     locationFeatDim=focusedCoordinateNum,
        #                     featureDim=self.config.featureDim)
        # self.alFusion = ALFusion(config=alConfig)
        # Create lstmExtractor object
        numHiddenUnits = self.config.featureDim
        # lstmConf = LSTMConfig(numHiddenUnits=numHiddenUnits,
        #                       visualFeatDim=self.config.featureDim,
        #                       isTrain=self.config.isTrain,
        #                       fcKeepProb=self.config.keepProb,
        #                       lstmKeepProb=self.config.keepProb,
        #                       isDropout=True)
        # self.lstmExtractor = LSTMExtractor(config=lstmConf)
        # # Create focused-point prediction object
        # fpConf = FpConfig(featureDim=numHiddenUnits,
        #                   numHiddenUnits=256,
        #                   coordinateDim=self.config.focusedCoordinateNum,
        #                   isTrain=self.config.isTrain,
        #                   keepProb=self.config.keepProb,
        #                   isDropout=True)
        # self.fpPredictor = FpPredictor(config=fpConf)

        # # Create reward object for object detection task
        # rewardConf = RewordConfig(featureDim=numHiddenUnits,
        #                           numHiddenUnits=128,
        #                           objectDim=self.config.objectDim,
        #                           isTrain=self.config.isTrain,
        #                           keepProb=self.config.keepProb,
        #                           isDropout=True)
        # self.objectDetectionReward = ObjectDetectionReward(config=rewardConf)
        claConf = ClaConfig(featureDim=config.featureDim,
                            numCategory=config.numCategory,
                            isTrain=config.isTrain,
                            keepProb=config.keepProb,
                            isDropout=True)
        self.classifier = ClaAction(config=claConf)


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
        maxSampleNum=1000,
        testingSampleRatio=0.3)
    dataset = VOCDataset(config=config)
    dataset.readDataset()

    isTrain = False
    expName = 'VOC_CD_RAM_512'
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
                    maxTrainEpoch=200,
                    keepProb=0.8,
                    modelRelativeDir=os.path.join('output/RAM', expName))
    config.isDispAlways = False
    # with tf.device('/gpu:2'):
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
