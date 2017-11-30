# encoding: utf8

import os
import sys

from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset
from src.DLModel.LeNetModel import Config
from src.DLModel.LeNetModel import LeNetModel


def main():
    print sys.argv
    # Load dataset
    dataHomeDir = '/home/jielyu/Database/MnistScaledNoisedObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/MnistObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/Mnist-dataset'
    mnistConfig = MnistConfig(batchSize=256,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    isTrain = True
    expName = 'MSNO_CD_LeNet'
    # Create Config object
    modelRelativeDir = os.path.join('output/Exp/DL', expName)
    config = Config(isTrain=isTrain,
                    inputShape=[56, 56, 1],
                    batchSize=256,
                    maxTrainEpoch=100,
                    modelRelativeDir=modelRelativeDir)
    # Create ram object
    model = LeNetModel(config=config)
    model.setDataset(dataset)
    if config.isTrain:
        model.trainCDModel()

    else:
        model.testCDModel()

if __name__ == '__main__':
    main()

