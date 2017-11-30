# encoding: utf8

# Import system libraries
import os
import sys

from src.AttentionModel.ClassificationDetectionRAM \
    import ClassificationDetectionRAM as CDRAM
from src.AttentionModel.ClassificationDetectionRAM import Config
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset


class MSOCDRAM(CDRAM):

    def __init__(self, config=Config()):
        super(MSOCDRAM, self).__init__(config=config)


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
    isTrain = False
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
                    monteCarloSample=5,
                    maxTrainEpoch=120,
                    keepProb=0.9,
                    modelRelativeDir='output/RAM/MSO_CD_RAM')
    # Create ram object
    ram = MSOCDRAM(config=config)
    ram.setDataset(dataset)
    if isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/saveTrack/MSO_CD_RAM/')
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)


if __name__ == '__main__':
    main()
