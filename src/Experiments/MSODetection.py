# encoding: utf8

# Import system libraries
import os
import sys

from src.AttentionModel.ObjectDetectionRAM import Config
from src.AttentionModel.ObjectDetectionRAM import ObjectDetectionRAM as ODRAM
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset


class MSODetectionRAM(ODRAM):

    def __init__(self, config=Config()):
        super(MSODetectionRAM, self).__init__(config=config)


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
                       keepProb=0.9,
                       modelRelativeDir='output/RAM/MSO_D_RAM')

    # Create ram object
    ram = MSODetectionRAM(config=ramConfig)
    ram.setDataset(dataset)

    if ramConfig.isTrain:
        # Train model
        ram.trainModel()
    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack/MSO_D_RAM/')
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)


if __name__ == '__main__':
    main()
