# encoding: utf8

from SuperModelConfig import SuperModelConfig


class SuperRAMConfig(SuperModelConfig):

    def __init__(self,
                 isRandomInitial=True,
                 initialMinValue=-0.3,
                 initialMaxValue=0.3,
                 isAddContext=True,
                 nScale=3,
                 scaleFactor=1.5,
                 minScaleSize=64,
                 targetSize=64,
                 featureDim=256,
                 focusedCoordinateNum=2,
                 batchSize=64,
                 maxTimeStep=10,
                 isTrain=False,
                 keepProb=1.0,
                 monteCarloSample=1,
                 isSamplePoint=True,
                 samplingStd=0.2,
                 maxGradNorm=500,
                 startLearningRate=3e-3,
                 minLearningRate=1e-4,
                 learningStepFactor=0.97,
                 maxTrainEpoch=500,
                 modelFileName='RAM.ckpt',
                 homeDir='/home/jielyu/Workspace/Python/AttentionModel',
                 modelRelativeDir='output/RAM/',
                 logFileName='RAM.log'):
        super(SuperRAMConfig, self).__init__(
            isTrain=isTrain,
            batchSize=batchSize,
            maxTrainEpoch=maxTrainEpoch,
            startLearningRate=startLearningRate,
            minLearningRate=minLearningRate,
            maxGradNorm=maxGradNorm,
            learningStepFactor=learningStepFactor,
            homeDir=homeDir,
            modelRelativeDir=modelRelativeDir,
            modelFileName=modelFileName,
            logFileName=logFileName)

        # Initially random flag
        self.isRandomInitial = isRandomInitial
        self.initialMinValue = initialMinValue
        self.initialMaxValue = initialMaxValue
        # Enable background
        self.isAddContext = isAddContext
        # The number of scales
        self.nScale = nScale
        # The factor of scale
        self.scaleFactor = scaleFactor
        # The size of the region with the minimum scale
        self.minScaleSize = minScaleSize
        # The size of the region fed into CNN
        self.targetSize = targetSize

        # The dimension of CNN and LSTM features
        self.featureDim = featureDim
        # The dimension of focused point
        self.focusedCoordinateNum = focusedCoordinateNum
        # The maximum steps in temporal
        self.maxTimeStep = maxTimeStep
        # The probability of keeping effectiveness in dropout
        self.keepProb = keepProb
        # Parameter of Monte Carlo samples
        self.monteCarloSample = monteCarloSample
        self.isSamplePoint = isSamplePoint
        self.samplingStd = samplingStd

    def saveTo(self, fid):
        super(SuperRAMConfig, self).saveTo(fid=fid)

        fid.write('\r\n')
        fid.write('Configuration for Object detection task:\r\n')
        fid.write('isRandomInitial = %s\r\n' % (self.isRandomInitial))
        fid.write('initialMinValue = %s\r\n' % (self.initialMinValue))
        fid.write('initialMaxValue = %s\r\n' % (self.initialMaxValue))
        fid.write('nScale = %s\r\n' % (self.nScale))
        fid.write('isAddContext = %s\r\n' % (self.isAddContext))
        fid.write('scaleFactor = %s\r\n' % (self.scaleFactor))
        fid.write('minScaleSize = %s\r\n' % (self.minScaleSize))
        fid.write('targetSize = %s\r\n' % (self.targetSize))
        fid.write('featureDim = %s\r\n' % (self.featureDim))
        fid.write('focusedCoordinateNum = %s\r\n' % (self.focusedCoordinateNum))
        fid.write('maxTimeStep = %s\r\n' % (self.maxTimeStep))
        fid.write('keepProb = %s\r\n' % (self.keepProb))
        fid.write('monteCarloSample = %s\r\n' % (self.monteCarloSample))
        fid.write('isSamplePoint = %s\r\n' % (self.isSamplePoint))
        fid.write('samplingStd = %s\r\n' % (self.samplingStd))
        fid.write('\r\n')
