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

from src.tools.Utils import Utils
from src.SuperClass.SuperModelConfig import SuperModelConfig as SuperConfig
from src.SuperClass.SuperModel import SuperModel
from src.AttentionModel.RAMEvaluator import RAMEvaluator

class Config(SuperConfig):

    def __init__(self,
                 isTrain = False,
                 startLearningRate=1e-4,
                 minLearningRate=1e-4,
                 maxTrainEpoch=100,
                 batchSize=64,
                 inputShape=[28, 28, 1],
                 numCategory=10,
                 homeDir='/home/jielyu/Workspace/Python/AttentionModel',
                 modelRelativeDir='output/DLM/LeNet/',
                 modelFileName='lenet.ckpt',
                 initialModelDir='',
                 logFileName='lenet.log'):

        super(Config, self).__init__(
            isTrain=isTrain,
            batchSize=batchSize,
            maxTrainEpoch=maxTrainEpoch,
            startLearningRate=startLearningRate,
            minLearningRate=minLearningRate,
            maxGradNorm=500,
            learningStepFactor=0.97,
            homeDir=homeDir,
            modelRelativeDir=modelRelativeDir,
            modelFileName=modelFileName,
            logFileName=logFileName)

        self.isDropout = True
        self.keepProb = 0.8
        self.initialModelDir = initialModelDir

        self.inputShape = inputShape
        self.numCategory = numCategory
        # Conv1
        self.w1Shape = [3, 3, self.inputShape[2], 16]
        self.b1Shape = [self.w1Shape[3]]
        self.pool1Shape = [1, 2, 2, 1]
        # Conv2
        self.w2Shape = [3, 3, self.w1Shape[3], 12]
        self.b2Shape = [self.w2Shape[3]]
        self.pool2Shape = [1, 2, 2, 1]
        # FC1
        shrinkingTimes = self.pool1Shape[1]*self.pool1Shape[2]\
                         *self.pool2Shape[1]*self.pool2Shape[1]
        inputDim = self.inputShape[0]*self.inputShape[1]
        numInput = inputDim/shrinkingTimes*self.w2Shape[3]
        self.w_fc1Shape = [numInput, 128]
        self.b_fc1Shape = [self.w_fc1Shape[1]]
        # FC2
        self.w_fc2Shape = [self.w_fc1Shape[1], self.numCategory]
        self.b_fc2Shape = [self.w_fc2Shape[1]]

    def saveTo(self, fid):
        super(Config, self).saveTo(fid)
        fid.write('inputShape = %s\r\n' % (self.inputShape))
        fid.write('numCategory = %s\r\n' % (self.numCategory))
        fid.write('initialModelDir = %s\r\n' % (self.initialModelDir))


class LeNetModel(SuperModel):

    def __init__(self, config=Config()):
        super(LeNetModel, self).__init__(config=config)

        inputShape = [None,
                      config.inputShape[0],
                      config.inputShape[1],
                      config.inputShape[2]]
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=inputShape)
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, config.numCategory])
        self.objects = tf.placeholder(dtype=tf.float32,
                                     shape=[None, 1, 4])
        self.conv1 = 0
        self.w1 = Utils.getTFVariable(name='w1',
                                      shape=config.w1Shape)
        self.b1 = Utils.getTFVariable(name='b1',
                                      shape=config.b1Shape)
        self.pool1 = 0
        self.conv2 = 0
        self.w2 = Utils.getTFVariable(name='w2',
                                      shape=config.w2Shape)
        self.b2 = Utils.getTFVariable(name='b2',
                                      shape=config.b2Shape)
        self.pool2 = 0
        self.fc1 = 0
        self.w_fc1 = Utils.getTFVariable(name='w_fc1',
                                         shape=config.w_fc1Shape)
        self.b_fc1 = Utils.getTFVariable(name='b_fc1',
                                         shape=config.b_fc1Shape)
        self.fc2 = 0
        self.w_fc2 = Utils.getTFVariable(name='w_fc2',
                                         shape=config.w_fc2Shape)
        self.b_fc2 = Utils.getTFVariable(name='b_fc2',
                                         shape=config.b_fc2Shape)
        self.preProb = 0
        self.acc = 0

        self.w_fc_yx = Utils.getTFVariable(name='w_fc_yx',
                                           shape=[config.w_fc1Shape[1], 2])
        self.b_fc_yx = Utils.getTFVariable(name='b_fc_yx', shape=[2])
        self.w_fc_hw = Utils.getTFVariable(name='w_fc_hw',
                                           shape=[config.w_fc1Shape[1],
                                                  2])
        self.b_fc_hw = Utils.getTFVariable(name='b_fc_hw',
                                           shape=[2])
        self.w_fc_score = \
            Utils.getTFVariable(name='w_fc_score',
                                shape=[config.w_fc1Shape[1], 1])
        self.b_fc_score = Utils.getTFVariable(name='b_fc_score',
                                              shape=[1])

        # Limit range
        self.yx = 0
        self.hw = 0
        self.score = 0

    def buildGraph(self, name='LeNetModel'):

        with tf.variable_scope(name):

            self.conv1 = tf.nn.conv2d(input=self.images,
                                      filter=self.w1,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      name='conv1') \
                         + self.b1
            self.pool1 = tf.nn.max_pool(value=self.conv1,
                                        ksize=self.config.pool1Shape,
                                        strides=self.config.pool1Shape,
                                        padding='SAME',
                                        name='pool1')
            self.conv2 = tf.nn.conv2d(input=self.pool1,
                                      filter=self.w2,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      name='conv2') \
                         + self.b2
            self.pool2 = tf.nn.max_pool(value=self.conv2,
                                        ksize=self.config.pool2Shape,
                                        strides=self.config.pool2Shape,
                                        padding='SAME',
                                        name='pool2')

            self.vec = tf.reshape(tensor=self.pool2,
                                  shape=[-1, self.config.w_fc1Shape[0]])

            self.fc1 = tf.nn.relu(
                features=tf.matmul(a=self.vec, b=self.w_fc1)
                         + self.b_fc1,
                name='fc1')

            if self.config.isTrain and self.config.isDropout:
                self.fc1 = tf.nn.dropout(
                    x=self.fc1,
                    keep_prob=self.config.keepProb)
            # Classification
            self.fc2 = tf.matmul(a=self.fc1, b=self.w_fc2) + self.b_fc2
            self.preProb = tf.nn.softmax(self.fc2, name='preProb')

            # Object regression
            self.yx = tf.matmul(self.fc1, self.w_fc_yx) + self.b_fc_yx
            self.hw = \
                tf.matmul(self.fc1, self.w_fc_hw) + self.b_fc_hw
            self.score = \
                tf.matmul(self.fc1, self.w_fc_score) + self.b_fc_score

            # Limitation if value range
            self.yx = tf.clip_by_value(self.yx, -1.0, 1.0)  # [-1,1]
            self.hw = tf.clip_by_value(self.hw, 0.0, 2.0)  # [0,2]
            self.score = tf.clip_by_value(self.score, 0.0, 1.0)  # [0, 1]
            self.objList = [[self.yx], [self.hw], [self.score]]
            self.preLabelList = [tf.argmax(self.preProb, 1)]

    def buildLossGraph(self, name='LossFunction'):
        gtCla = self.labels
        preCla = self.preProb
        gtYXHW = self.objects
        with tf.variable_scope(name):
            # Classification loss
            ce = tf.reduce_mean(
                -tf.reduce_sum(
                    gtCla * tf.log(tf.clip_by_value(
                        preCla, 1e-10, 1.0)), reduction_indices=[1]))
            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        x=tf.argmax(preCla, 1),
                        y=tf.argmax(gtCla, 1)),
                    tf.float32))
            self.ce = ce
            self.acc = acc

            # Detection
            gtYX = gtYXHW[:, 0, 0:2]
            preYX = self.yx
            ds = tf.reduce_sum(
                tf.square(gtYX - preYX),
                reduction_indices=1
            )
            ds = tf.reduce_mean(ds)

            preHW = self.hw
            msr, overlap = self.buildOverlapGraph(preYX=preYX,
                                                  preHW=preHW,
                                                  gtYXHW=gtYXHW)
            msr = tf.reduce_mean(msr)
            overlap = tf.reduce_mean(overlap)+1e-8
            # Loss
            self.loss = ce + ds - tf.log(overlap)

            self.msr = msr
            self.overlap = overlap

    def buildOverlapGraph(self, preYX, preHW, gtYXHW, name='Overlap'):
        with tf.variable_scope(name):
            gtYX = gtYXHW[:, 0, 0:2]
            gtHW = gtYXHW[:, 0, 2:4]
            # Get timestep

            # Compute mean square root of center point
            msr = tf.sqrt(
                tf.reduce_sum(
                    tf.square((gtYX + gtHW / 2.0) -
                              (preYX + preHW / 2.0)),
                    reduction_indices=1
                )
            )

            # Compute overlap
            # dy = max(min(yp1, yt1) - max(yp0, yt0), 0)
            # dx = max(min(xp1, xt1) - max(xp0, xt0), 0)
            delta_y = tf.nn.relu(
                tf.reduce_min(
                    tf.concat(concat_dim=1,
                              values=[tf.reshape(preYX[:, 0]+preHW[:, 0],
                                                 shape=[-1, 1]),
                                      tf.reshape(gtYX[:, 0] + gtHW[:, 0],
                                                 shape=[-1, 1])]),
                    reduction_indices=1) -
                tf.reduce_max(
                    tf.concat(concat_dim=1,
                              values=[tf.reshape(preYX[:, 0],
                                                 shape=[-1, 1]),
                                      tf.reshape(gtYX[:, 0],
                                                 shape=[-1, 1])]),
                    reduction_indices=1)
            )
            delta_x = tf.nn.relu(
                tf.reduce_min(
                    tf.concat(concat_dim=1,
                              values=[tf.reshape(preYX[:, 1]+preHW[:, 1],
                                                 shape=[-1, 1]),
                                      tf.reshape(gtYX[:, 1] + gtHW[:, 1],
                                                 shape=[-1, 1])]),
                    reduction_indices=1) -
                tf.reduce_max(
                    tf.concat(concat_dim=1,
                              values=[tf.reshape(preYX[:, 1],
                                                 shape=[-1, 1]),
                                      tf.reshape(gtYX[:, 1],
                                                 shape=[-1, 1])]),
                    reduction_indices=1)
            )
            # a = dy*dx
            # r = a/(ga+pa-a)
            overArea = delta_y * delta_x
            preArea = preHW[:, 0] * preHW[:, 1]
            gtArea = gtHW[:, 0] * gtHW[:, 1]
            overlap = overArea / (gtArea + preArea - overArea)

        return msr, overlap

    def trainModel(self, model='', sess=None):
        # Create session
        sess = self.createSession()

        lossTrainList = []
        accTrainList = []
        lossTestList = []
        accTestList = []
        for i in range(0, self.config.maxTrainTimes):

            epochCnt = i // self.config.minDispStep
            images, bbox, labels \
                = self.dataset.trainset.getNextBatchWithLabels()
            bbox = Utils.convertToYXHW(bbox)
            images, bbox = Utils.normalizeImagesAndBbox(images, bbox)
            feed_dict = {self.images: images, self.labels: labels}
            # Train
            sess.run(self.trainOp, feed_dict=feed_dict)

            # Test
            if i == 0 \
                    or i % self.config.minDispStep == self.config.minDispStep - 1 \
                    or self.config.isDispAlways:
                # Test on trainset
                loss, acc = sess.run((self.loss, self.acc),
                                     feed_dict=feed_dict)
                # Display information
                t_str = 'step=%d ' % (i) + \
                        'epoch=%d ' % (epochCnt) + \
                        'loss=%f ' % (loss) + \
                        'acc=%f ' % (acc)
                lossTrainList.append(loss)
                accTrainList.append(acc)
                self.printLog(t_str)

                # Test and save
                if i == 0 \
                        or i % self.config.minDispStep == self.config.minDispStep - 1:
                    # Save model
                    Utils.saveModel(sess=sess,
                                    modelDir=self.config.modelDir,
                                    modelFileName=self.config.modelFileName)
                    # Test on testset
                    teLoss, teAcc = self.testModel(sess=sess)
                    lossTestList.append(teLoss)
                    accTestList.append(teAcc)
        self.printLog('Train completely')
        self.saveTrainingCurve(scalarList=lossTrainList, name='train_loss')
        self.saveTrainingCurve(scalarList=accTrainList, name='train_acc')
        self.saveTrainingCurve(scalarList=lossTestList, name='test_loss')
        self.saveTrainingCurve(scalarList=accTestList, name='test_acc')

    def trainCDModel(self, modelDir='', sess=None):
        # Check training flag
        if not self.config.isTrain:
            raise ValueError('Not Training Configuration')
        # Check modelDir
        if not os.path.isdir(modelDir):
            modelDir = self.config.modelDir

        if sess is None:
            sess = self.createSession()
        # Load initialize model
        if os.path.isdir(self.config.initialModelDir):
            Utils.loadModel(sess=sess,
                            modelDir=self.config.initialModelDir,
                            modelFileName=self.config.modelFileName)
        # Train model
        self.printLog('start to train RAM')

        t_str = ': trainTimes=%s' % (self.config.maxTrainTimes)
        self.printLog(t_str)
        t_str = ': dispTimes=%s' % (self.config.minDispStep)
        self.printLog(t_str)

        lossTrainList = []
        overlapTrainList = []
        accTrainList = []
        lossTestList = []
        overlapTestList = []
        accTestList = []
        for i in range(0, self.config.maxTrainTimes):

            epochCnt = i // self.config.minDispStep
            # Obtain a batch of data
            images, bbox, labels = \
                self.dataset.trainset.getNextBatchWithLabels()
            bbox = Utils.convertToYXHW(bbox)
            images, bbox = Utils.normalizeImagesAndBbox(images, bbox)
            feed_dict = {self.images: images,
                         self.objects: bbox,
                         self.labels: labels}
            # Train
            sess.run(self.trainOp, feed_dict=feed_dict)

            # Test
            if i == 0 \
                    or i % self.config.minDispStep == \
                                    self.config.minDispStep - 1 \
                    or self.config.isDispAlways:
                # Test on trainset
                loss, overlap, msr, ce, acc, lr = \
                    sess.run((self.loss, self.overlap, self.msr,
                              self.ce, self.acc, self.learningRate),
                             feed_dict=feed_dict)
                # Display information
                t_str = 'step=%d ' % (i) + \
                        'epoch=%d/%d ' % (epochCnt,
                                          self.config.maxTrainEpoch) + \
                        'loss=%f ' % (loss) + \
                        'ol=%f ' % (overlap) + \
                        'msr=%f ' % (msr) + \
                        'ce=%f ' % (ce) + \
                        'acc=%f ' % (acc) + \
                        'lr=%f ' % (lr)
                lossTrainList.append(loss)
                overlapTrainList.append(overlap)
                accTrainList.append(acc)
                self.printLog(t_str)

                # Test and save
                if i == 0 \
                        or i % self.config.minDispStep == \
                                        self.config.minDispStep - 1:
                    # Save model
                    Utils.saveModel(sess=sess,
                                    modelDir=modelDir,
                                    modelFileName=self.config.modelFileName)
                    # Test on testset
                    teLoss, teOverlap, teAcc = self.testCDModel(sess=sess)

                    lossTestList.append(teLoss)
                    overlapTestList.append(teOverlap)
                    accTestList.append(teAcc)
        self.printLog('Train completely')

        # Draw Training curve
        self.saveTrainingCurve(scalarList=lossTrainList,
                               name='train_loss')
        self.saveTrainingCurve(scalarList=overlapTrainList,
                               name='train_overlap')
        self.saveTrainingCurve(scalarList=accTrainList,
                               name='train_acc')
        self.saveTrainingCurve(scalarList=lossTestList,
                               name='test_loss')
        self.saveTrainingCurve(scalarList=overlapTestList,
                               name='test_overlap')
        self.saveTrainingCurve(scalarList=accTestList,
                               name='test_acc')

    def testModel(self, modelDir='', sess=None):
        if sess is None:
            sess = self.createSession()
            # Load model
            if not self.config.isTrain:
                Utils.loadModel(sess=sess,
                                modelDir=self.config.modelDir,
                                modelFileName=self.config.modelFileName)
        # Test on testset
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumAcc = 0
        testSampleNum = \
            numTestSample // self.config.batchSize
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox, t_labels = \
                self.dataset.testset.getNextBatchWithLabels()
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.labels: t_labels}
            # Obtain accuracy
            loss, acc = \
                sess.run((self.loss, self.acc),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumAcc += acc / testSampleNum
        t_str = 'testing ' \
                'acc=%f ' % (sumAcc)
        self.printLog(t_str)
        # Return
        return sumLoss, sumAcc

    def testCDModel(self,
                    modelDir='',
                    testset=None,
                    sess=None):
        print 'testing on the testset ... '
        # Check model direction
        if not os.path.isdir(modelDir):
            modelDir = self.config.modelDir

        if sess is None:
            sess = self.createSession()
            # Load model
            if not self.config.isTrain:
                Utils.loadModel(sess=sess,
                                modelDir=modelDir,
                                modelFileName=self.config.modelFileName)

        # Test on testset
        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumOverlap = 0
        sumMsr = 0
        sumAcc = 0
        testSampleNum = \
            (numTestSample // self.config.batchSize) + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox, t_labels = \
                self.dataset.testset.getNextBatchWithLabels()
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.objects: t_bbox,
                         self.labels: t_labels}
            # Obtain accuracy
            loss, overlap, msr, ce, acc = \
                sess.run((self.loss,
                          self.overlap, self.msr, self.ce, self.acc),
                         feed_dict=feed_dict)
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            sumMsr += msr / testSampleNum
            sumAcc += acc / testSampleNum
        t_str = 'testing overlap=%f ' % (sumOverlap) + \
                'msr=%f ' % (sumMsr) + \
                'acc=%f ' % (sumAcc)
        self.printLog(t_str)
        # Return
        return sumLoss, sumOverlap, sumAcc

    def evaluate(self,
                 modelDir='',
                 sess=None,
                 saveDir=None,
                 saveMaxNum=100,
                 isSaveSeq=False,
                 ext='png'):
        print 'evaluating on the testset ... '
        # Check model direction
        if not os.path.isdir(modelDir):
            modelDir = self.config.modelDir

        if sess is None:
            sess = self.createSession()
            # Load model
            if not self.config.isTrain:
                Utils.loadModel(sess=sess,
                                modelDir=modelDir,
                                modelFileName=self.config.modelFileName)

        # Test on testset
        imagesList = []
        gtLabelsList = []
        gtBboxesList = []
        preYXListList = []
        preHWListList = []
        preScoreListList = []
        preLabelsListList = []

        numTestSample = self.dataset.testset.getSampleNum()
        sumLoss = 0
        sumOverlap = 0
        sumMsr = 0
        sumAcc = 0
        testSampleNum = \
            (numTestSample // self.config.batchSize) + 1
        for j in range(0, testSampleNum):
            # Get a batch samples
            t_images, t_bbox, t_labels = \
                self.dataset.testset.getNextBatchWithLabels()
            gtLabels = np.argmax(t_labels, axis=1)
            imagesList.append(t_images)
            gtBboxesList.append(t_bbox)
            gtLabelsList.append(gtLabels)
            # Convert to standard mode
            t_bbox = Utils.convertToYXHW(t_bbox)
            t_images, t_bbox = \
                Utils.normalizeImagesAndBbox(t_images, t_bbox)
            # Input interface
            feed_dict = {self.images: t_images,
                         self.objects: t_bbox,
                         self.labels: t_labels}
            # Run graph
            loss, overlap, msr, ce, \
            acc, objList, preLabelList = \
                sess.run((self.loss, self.overlap, self.msr, self.ce,
                          self.acc, self.objList, self.preLabelList),
                         feed_dict=feed_dict)
            preYXListList.append(objList[0])
            preHWListList.append(objList[1])
            preScoreListList.append(objList[2])
            preLabelsListList.append(preLabelList)
            # Accumulate
            sumLoss += loss / testSampleNum
            sumOverlap += overlap / testSampleNum
            sumMsr += msr / testSampleNum
            sumAcc += acc / testSampleNum
        # Construct dictionary
        evaluateDataDict = {
            'imagesList': imagesList,
            'gtLabelsList': gtLabelsList,
            'gtBboxesList': gtBboxesList,
            'preYXListList': preYXListList,
            'preHWListList': preHWListList,
            'preScoreListList': preScoreListList,
            'preLabelsListList': preLabelsListList}
        # Compute mAP and draw process
        items = RAMEvaluator.parseDict(evaluateDataDict)
        gtBboxes, preBboxes, gtLabels, preLabels = \
            RAMEvaluator.parseBboxesLabelsForSingleObj(items=items)
        mAP = RAMEvaluator.evaluate_mAP(
            gtBboxes, preBboxes, gtLabels, preLabels)
        if saveDir is not None:
            if not os.path.isdir(saveDir):
                os.makedirs(saveDir)
            RAMEvaluator.drawMultiProcess(
                items=items,
                indexList=range(0, min(saveMaxNum, len(items))),
                isSeq=isSaveSeq,
                saveDir=saveDir,
                ext=ext)

        # Display info
        t_str = 'evaluating overlap=%f ' % (sumOverlap) + \
                'msr=%f ' % (sumMsr) + \
                'acc=%f ' % (sumAcc) + \
                'mAP=%f ' % (mAP)
        self.printLog(t_str)


def main():
    print sys.argv
    # Load dataset
    dataHomeDir = '/home/jielyu/Database/MnistScaledNoisedObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/MnistScaledObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/MnistObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/Mnist-dataset'
    mnistConfig = MnistConfig(batchSize=256,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    isTrain = False
    expName = 'CD_LeNet'
    # Create Config object
    modelRelativeDir = os.path.join('output/DL', expName)
    config = Config(isTrain=isTrain,
                    inputShape=[56, 56, 1],
                    batchSize=256,
                    maxTrainEpoch=200,
                    modelRelativeDir=modelRelativeDir)
    config.initialModelDir = config.modelDir
    # Create ram object
    model = LeNetModel(config=config)
    model.setDataset(dataset)
    if config.isTrain:
        model.trainCDModel()

    else:
        # model.testCDModel()
        model.evaluate()

if __name__ == '__main__':
    main()
