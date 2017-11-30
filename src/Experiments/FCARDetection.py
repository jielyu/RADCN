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
from src.AttentionModel.Reward.ObjectDetectionReward \
    import Config as RewordConfig
from src.AttentionModel.Reward.ObjectDetectionReward \
    import ObjectDetectionReward
from src.Dataset.FCARDataset import Config as FCARConfig
from src.Dataset.FCARDataset import FCARDataset


class FCARDetectionRAM(ODRAM):

    def __init__(self, config=Config()):
        super(FCARDetectionRAM, self).__init__(config=config)

        w1_chNum = 32
        w2_chNum = 64
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
        lstmConf = LSTMConfig(numHiddenUnits=numHiddenUnits,
                              visualFeatDim=self.config.featureDim,
                              isTrain=self.config.isTrain,
                              fcKeepProb=self.config.keepProb,
                              lstmKeepProb=self.config.keepProb,
                              isDropout=True)
        self.lstmExtractor = LSTMExtractor(config=lstmConf)

        # Create focused-point prediction object
        fpConf = FpConfig(featureDim=numHiddenUnits,
                          numHiddenUnits=256,
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


def main():
    print sys.argv

    # Load dataset
    dataHomeDir = '/home/jielyu/Database/FCAR-dataset'
    fcarConfig = FCARConfig(
        enableMemSave=True,
        batchSize=64,
        datasetDir=dataHomeDir,
        maxSampleNum=1000)
    dataset = FCARDataset(config=fcarConfig)
    dataset.readDataset()

    # Create Config
    homeDir = '/home/jielyu/Workspace/Python/AttentionModel/'
    expName = 'FCAR_OD_RAM'
    ramConfig = Config(isTrain=False,
                       batchSize=64,
                       isAbsoluteAttention=True,
                       inputShape=[800, 800, 3],
                       nScale=3,
                       featureDim=512,
                       scaleFactor=4.5,
                       isAddContext=False,
                       minScaleSize=32,
                       targetSize=32,
                       startLearningRate=1e-4,
                       minLearningRate=1e-4,
                       learningStepFactor=0.99,
                       monteCarloSample=1,
                       maxTrainEpoch=40,
                       keepProb=0.8,
                       homeDir=homeDir,
                       modelRelativeDir=os.path.join('output/RAM', expName))
    # ramConfig.initialModelDir = os.path.join(homeDir, 'output/RAM/FCAR_OD_RAM')

    # Create ram object
    ram = FCARDetectionRAM(config=ramConfig)
    ram.setDataset(dataset)

    if ramConfig.isTrain:
        # Train model
        ram.trainModel()
    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack',
                                expName)
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)

if __name__ == '__main__':
    main()
