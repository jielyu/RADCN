# encoding: utf8

import os
import sys

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf

# Self-define libraries
from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset

from src.tools.Utils import Utils
from src.SuperClass.SuperModelConfig import SuperModelConfig as SuperConfig
from src.SuperClass.SuperModel import SuperModel


class Config(SuperConfig):

    def __init__(self,
                 isTrain = False,
                 startLearningRate=1e-3,
                 minLearningRate=1e-5,
                 maxTrainEpoch=100,
                 batchSize=64,
                 inputShape=[224, 224, 3],
                 numCategory=10,
                 homeDir='/home/jielyu/Workspace/Python/AttentionModel',
                 modelRelativeDir='output/DLM/AlexNet/',
                 modelFileName='alexnet.ckpt',
                 logFileName='alexnet.log'):
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
        self.inputShape = inputShape
        self.numCategory = numCategory
        self.keepProb = 0.99

        # Conv1
        # self.w1Shape = [11, 11, self.inputShape[2], 96]
        self.w1Shape = [3, 3, self.inputShape[2], 96]
        self.b1Shape = [self.w1Shape[3]]
        self.s1Shape = [1, 1, 1, 1]
        self.kp1Shape = [1, 3, 3, 1]
        self.sp1Shape = [1, 2, 2, 1]
        # Conv2
        self.w2Shape = [5, 5, self.w1Shape[3], 256]
        self.b2Shape = [self.w2Shape[3]]
        self.kp2Shape = [1, 3, 3, 1]
        self.sp2Shape = [1, 2, 2, 1]
        # Conv3
        self.w3Shape = [3, 3, self.w2Shape[3], 384]
        self.b3Shape = [self.w3Shape[3]]
        # Conv4
        self.w4Shape = [3, 3, self.w3Shape[3], 384]
        self.b4Shape = [self.w4Shape[3]]
        # Conv5
        self.w5Shape = [3, 3, self.w4Shape[3], 256]
        self.b5Shape = [self.w5Shape[3]]
        # fc1
        shkFactor = self.sp1Shape[1]*self.sp1Shape[2] * \
                    self.sp2Shape[1]*self.sp2Shape[2] * \
                    self.s1Shape[1]*self.s1Shape[2]
        inputDim = self.inputShape[0]*self.inputShape[1]
        numInput = inputDim/shkFactor*self.w5Shape[3]
        self.w_fc1Shape = [numInput, 4096]
        self.b_fc1Shape = [self.w_fc1Shape[1]]
        # fc2
        self.w_fc2Shape = [self.w_fc1Shape[1], 4096]
        self.b_fc2Shape = [self.w_fc2Shape[1]]
        # fc3
        self.w_fc3Shape = [self.w_fc2Shape[1], self.numCategory]
        self.b_fc3Shape = [self.w_fc3Shape[1]]

    def saveTo(self, fid):
        super(Config, self).saveTo(fid)
        fid.write('inputShape = %s\r\n' % (self.inputShape))
        fid.write('numCategory = %s\r\n' % (self.numCategory))


class AlexNetModel(SuperModel):

    def __init__(self, config=Config()):
        super(AlexNetModel, self).__init__(config=config)

        inputShape = [None,
                      config.inputShape[0],
                      config.inputShape[1],
                      config.inputShape[2]]
        self.images = tf.placeholder(dtype=tf.float32,
                                     shape=inputShape,
                                     name='images')
        self.labels = tf.placeholder(dtype=tf.float32,
                                     shape=[None, config.numCategory],
                                     name='labels')
        # Conv1
        self.w1 = Utils.getTFVariable(name='w1', shape=self.config.w1Shape)
        self.b1 = Utils.getTFVariable(name='b1', shape=self.config.b1Shape)

        # Conv2
        self.w2 = Utils.getTFVariable(name='w2', shape=self.config.w2Shape)
        self.b2 = Utils.getTFVariable(name='b2', shape=self.config.b2Shape)

        # Conv3
        self.w3 = Utils.getTFVariable(name='w3', shape=self.config.w3Shape)
        self.b3 = Utils.getTFVariable(name='b3', shape=self.config.b3Shape)

        # Conv4
        self.w4 = Utils.getTFVariable(name='w4', shape=self.config.w4Shape)
        self.b4 = Utils.getTFVariable(name='b4', shape=self.config.b4Shape)

        # Conv5l
        self.w5 = Utils.getTFVariable(name='w5', shape=self.config.w5Shape)
        self.b5 = Utils.getTFVariable(name='b5', shape=self.config.b5Shape)

        # FC1
        self.w_fc1 = Utils.getTFVariable(name='w_fc1',
                                         shape=self.config.w_fc1Shape)
        self.b_fc1 = Utils.getTFVariable(name='b_fc1',
                                         shape=self.config.b_fc1Shape)

        # FC2
        self.w_fc2 = Utils.getTFVariable(name='w_fc2',
                                         shape=self.config.w_fc2Shape)
        self.b_fc2 = Utils.getTFVariable(name='b_fc2',
                                         shape=self.config.b_fc2Shape)

        # FC3
        self.w_fc3 = Utils.getTFVariable(name='w_fc3',
                                         shape=self.config.w_fc3Shape)
        self.b_fc3 = Utils.getTFVariable(name='b_fc3',
                                         shape=self.config.b_fc3Shape)

    def buildGraph(self, name='AlexNet'):

        with tf.variable_scope(name):

            # Conv1
            self.conv1 = tf.nn.relu(
                features=tf.nn.conv2d(input=self.images,
                                      filter=self.w1,
                                      strides=self.config.s1Shape,
                                      padding='SAME')
                         + self.b1,
                name='conv1')
            self.pool1 = tf.nn.max_pool(value=self.conv1,
                                        ksize=self.config.kp1Shape,
                                        strides=self.config.sp1Shape,
                                        padding='SAME',
                                        name='pool1')
            self.lrn1 = tf.nn.lrn(self.pool1, 4,
                                  bias=1.0,
                                  alpha=0.001 / 9.0,
                                  beta=0.75,
                                  name='lrn2')
            if self.config.isTrain:
                self.lrn1 = tf.nn.dropout(self.lrn1,
                                          keep_prob=self.config.keepProb)

            # Conv2
            self.conv2 = tf.nn.relu(
                features=tf.nn.conv2d(input=self.lrn1,
                                      filter=self.w2,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + self.b2,
                name='conv2')
            self.pool2 = tf.nn.max_pool(value=self.conv2,
                                        ksize=self.config.kp2Shape,
                                        strides=self.config.sp2Shape,
                                        padding='SAME',
                                        name='pool2')
            self.lrn2 = tf.nn.lrn(self.pool2, 4,
                                  bias=1.0,
                                  alpha=0.001/9.0,
                                  beta=0.75,
                                  name='lrn2')
            if self.config.isTrain:
                self.lrn2 = tf.nn.dropout(self.lrn2,
                                          keep_prob=self.config.keepProb)

            # Conv3
            self.conv3 = tf.nn.relu(
                features=tf.nn.conv2d(input=self.lrn2,
                                      filter=self.w3,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + self.b3,
                name='conv3')
            # Conv4
            self.conv4 = tf.nn.relu(
                features=tf.nn.conv2d(input=self.conv3,
                                      filter=self.w4,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + self.b4,
                name='conv4')
            # Conv5
            self.conv5 = tf.nn.relu(
                features=tf.nn.conv2d(input=self.conv4,
                                      filter=self.w5,
                                      strides=[1, 1, 1, 1],
                                      padding='SAME')
                         + self.b5,
                name='conv5')
            # FC1
            self.vec = tf.reshape(self.conv5,
                                  shape=[-1, self.config.w_fc1Shape[0]])
            self.fc1 = tf.nn.relu(
                features=tf.matmul(a=self.vec, b=self.w_fc1)
                                           + self.b_fc1,
                name='fc1')
            self.fc2 = tf.nn.relu(
                features=tf.matmul(a=self.fc1, b=self.w_fc2)
                         + self.b_fc2,
                name='fc2')
            # self.fc3 = tf.nn.relu(
            #     features=tf.matmul(a=self.fc2, b=self.w_fc3)
            #              + self.b_fc3,
            #     name='fc3')
            self.fc3 = tf.matmul(a=self.fc2, b=self.w_fc3) + self.b_fc3,
            print self.fc3
            # self.preCla = tf.nn.softmax(self.fc3, name='preCla')
            self.preCla = tf.squeeze(tf.nn.softmax(self.fc3))

    def buildLossGraph(self):

        gtCla = self.labels
        preCla = self.preCla
        # print 'gtCla = ', gtCla
        # print 'preCla = ', preCla
        with tf.variable_scope('LossGraph'):
            ce = tf.reduce_mean(
                -tf.reduce_sum(
                    gtCla * tf.log(tf.clip_by_value(
                        preCla,
                        1e-10,
                        1.0)),
                    reduction_indices=[1]))
            acc = tf.reduce_mean(
                tf.cast(
                    tf.equal(
                        x=tf.argmax(preCla, 1),
                        y=tf.argmax(gtCla, 1)),
                    tf.float32))
            self.loss = ce
            self.acc = acc

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

            # output, gt = \
            # sess.run((self.preCla, self.labels), feed_dict=feed_dict)
            # print 'output = ', output
            # print 'sum = ', np.sum(output, axis=1)
            # print 'gt = ', gt

            # Test
            if i == 0 \
                    or i % self.config.minDispStep == \
                                    self.config.minDispStep - 1 \
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
                        or i % self.config.minDispStep == \
                                        self.config.minDispStep - 1:
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


def main():
    print sys.argv
    # Load dataset
    dataHomeDir = '/home/jielyu/Database/MnistScaledObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/MnistObject-dataset'
    # dataHomeDir = '/home/jielyu/Database/Mnist-dataset'
    mnistConfig = MnistConfig(batchSize=256,
                              datasetDir=dataHomeDir,
                              maxSampleNum=100000,
                              testingSampleRatio=0.3)
    dataset = MnistObjectDataset(config=mnistConfig)
    dataset.readDataset()

    with tf.device('/gpu:1'):
        isTrain = True
        # Create Config object
        config = Config(isTrain=isTrain,
                        inputShape=[56, 56, 1],
                        batchSize=256,
                        maxTrainEpoch=100)
        # Create ram object
        model = AlexNetModel(config=config)
        model.setDataset(dataset)
        if config.isTrain:
            model.trainModel()

        else:
            model.testModel()

if __name__ == '__main__':
    main()
