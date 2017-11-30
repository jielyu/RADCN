# encoding: utf8

import os
import sys

from src.tools.Utils import Utils


class SuperModelConfig(object):

    def __init__(self,
                 isTrain=False,
                 batchSize=64,
                 maxTrainEpoch=100,
                 startLearningRate=1e-4,
                 minLearningRate='1e-4',
                 maxGradNorm=500,
                 learningStepFactor=0.97,
                 homeDir='/home/jielyu/Workspace/Python/AttentionModel',
                 modelRelativeDir='output/Debug/TFModel',
                 modelFileName='model.ckpt',
                 logFileName='model.log'):
        # Direction configuration
        self.homeDir = homeDir
        self.modelRelativeDir = modelRelativeDir
        self.modelFileName = modelFileName
        self.modelDir = os.path.join(self.homeDir, self.modelRelativeDir)
        self.logDir = self.modelDir
        self.logFileName = Utils.getTimeStamp() + '-' + logFileName

        self.isTrain = isTrain
        self.startLearningRate = startLearningRate
        self.minLearningRate = minLearningRate
        self.maxTrainEpoch = maxTrainEpoch
        self.maxGradNorm = maxGradNorm
        self.learningStepFactor = learningStepFactor

        self.batchSize = batchSize
        self.minDispStep = 1
        self.maxTrainTimes = 100
        self.isDispAlways = True
        # Enable whether using all memory
        self.isFullMemory = False

    def saveTo(self, fid):
        if not isinstance(fid, file):
            raise TypeError('Not file object')
        fid.write('\r\n')
        fid.write('Configuration of Parameters for Running Program\r\n')
        fid.write('homeDir = %s\r\n' % (self.homeDir))
        fid.write('modelDir = %s\r\n' % (self.modelDir))
        fid.write('modelFileName = %s\r\n' % (self.modelFileName))
        fid.write('logDir = %s\r\n' % (self.logDir))
        fid.write('logFileName = %s\r\n' % (self.logFileName))
        fid.write('isTrain = %s\r\n' % (self.isTrain))
        fid.write('startLearningRate = %s\r\n' % (self.startLearningRate))
        fid.write('minLearningRate = %s\r\n' % (self.minLearningRate))
        fid.write('maxTrainEpoch = %s\r\n' % (self.maxTrainEpoch))
        fid.write('maxGradNorm = %s\r\n' % (self.maxGradNorm))
        fid.write('learningStepFactor = %s\r\n' % (self.learningStepFactor))
        fid.write('batchSize = %s\r\n' % (self.batchSize))


def main():
    print sys.argv

    config = SuperModelConfig()
    if not os.path.isdir(config.modelDir):
        os.makedirs(config.modelDir)
    with open(os.path.join(config.logDir, config.logFileName), 'w') as fid:
        config.saveTo(fid)

if __name__ == '__main__':
    main()
