# encoding: utf8
# Standard libraries
import os
import sys
# 3rd part libraries
import tensorflow as tf
import numpy as np
# Self-define libraries
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset

from src.DLModel.LeNetModel import Config
from src.DLModel.LeNetModel import LeNetModel


def main():
    print sys.argv
    # Load dataset
    dataHomeDir = '/home/jielyu/Database/ClutteredTranslatedMnist100x100-dataset'
    mnistConfig = MnistConfig(batchSize=64,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    isTrain = True
    expName = 'exp_ct100_CD_LeNet'
    # Create Config object
    modelRelativeDir = os.path.join('output/Exp/DL', expName)
    config = Config(isTrain=isTrain,
                    inputShape=[100, 100, 1],
                    batchSize=64,
                    maxTrainEpoch=100,
                    startLearningRate=1e-4,
                    minLearningRate=1e-4,
                    modelRelativeDir=modelRelativeDir)
    config.initialModelDir = config.modelDir
    # Create ram object
    model = LeNetModel(config=config)
    model.setDataset(dataset)
    if config.isTrain:
        model.trainCDModel()

    else:
        # model.testCDModel()
        trackDir = os.path.join(model.config.homeDir,
                                'output/Exp/saveTrack',
                                expName)
        # model.evaluate()
        model.evaluate(saveDir=trackDir,
                       saveMaxNum=100,
                       isSaveSeq=False,
                       ext='png')

if __name__ == '__main__':
    main()
