# encoding: utf8

# Import system libraries
import os
import sys
# Import self-define libraries
from src.AttentionModel.ClassificationRAM import ClassificationRAM as ClaRAM
from src.AttentionModel.ClassificationRAM import Config
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset


class MSNOClasRAM(ClaRAM):
    def __init__(self, config=Config()):
        super(MSNOClasRAM, self).__init__(config=config)


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
                    monteCarloSample=3,
                    maxTrainEpoch=100,
                    keepProb=0.9,
                    modelRelativeDir='output/RAM/MSNO_C_RAM')
    # Create ram object
    ram = MSNOClasRAM(config=config)
    ram.setDataset(dataset)
    if config.isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(config.homeDir,
                                'output/saveTrack/MSNO_C_RAM/')
        ram.testModel(isSaveTrack=True,
                      saveDir=trackDir)


if __name__ == '__main__':
    main()
