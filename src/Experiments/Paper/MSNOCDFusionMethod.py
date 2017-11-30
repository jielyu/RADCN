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
    dataHomeDir = '/media/home_bak/jielyu/Database/MnistScaledNoisedObject-dataset'
    mnistConfig = MnistConfig(batchSize=64,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    # with tf.device('/gpu:0'):
    isTrain = False

    # Parameters
    initRange = 0.3
    nScale = 3
    scaleFactor = 2.0
    featDim = 256
    glimpStep = 10

    fusionMethod = 'app_only'  # 'plus', 'product', 'concat', 'app_only'
    # The name of exp
    expName = 'exp_msno' + '_fusion_' + fusionMethod

    # Create Config object
    config = Config(isTrain=isTrain,
                    inputShape=[56, 56, 1],
                    isRandomInitial=True,
                    initialMinValue=-initRange,
                    initialMaxValue=initRange,
                    featureDim=featDim,
                    nScale=nScale,
                    scaleFactor=scaleFactor,
                    isAddContext=False,
                    minScaleSize=8,
                    fusionMethod=fusionMethod,
                    maxTimeStep=glimpStep,
                    targetSize=16,
                    startLearningRate=1e-3,
                    minLearningRate=1e-4,
                    monteCarloSample=5,
                    maxTrainEpoch=120,
                    keepProb=0.9,
                    modelRelativeDir=os.path.join('output/Exp/RAM',
                                                  expName))
    # Create ram object
    ram = MSOCDRAM(config=config)
    ram.setDataset(dataset)
    if isTrain:
        ram.trainModel()

    else:
        trackDir = os.path.join(ram.config.homeDir,
                                'output/Exp/saveTrack',
                                expName)
        # ram.testModel(isSaveTrack=True,
        #               saveDir=trackDir)

        # ram.testModel()
        # Evaluate
        ram.evaluate()


if __name__ == '__main__':
    main()
