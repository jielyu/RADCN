# encoding: utf8

import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from SuperModelConfig import SuperModelConfig as Config
from src.tools.Utils import Utils


class SuperModel(object):
    """
    Function: Package common operations and properties
    """

    def __init__(self, config=Config()):
        """
        Function: Initialize common properties and directions
        Note: This function should be overridden by subclass
        :param config:
        """
        self.config = config
        self.dataset = 0
        self.log = 0
        self.sess = 0
        self.output = 0
        self.loss = 0
        self.learningRate = 0
        self.trainOp = 0
        # Create model direction
        if not os.path.isdir(config.modelDir):
            os.makedirs(config.modelDir)
        # Create log object
        if os.path.isdir(config.logDir):
            logPath = os.path.join(config.logDir, config.logFileName)
            self.log = open(logPath, 'w')
            self.log.write('Log for Tensorflow Program\r\n')
            self.config.saveTo(self.log)

    def __del__(self):
        # Close log object
        if not isinstance(self.log, int):
            self.log.close()
        # Close session
        if not isinstance(self.sess, int):
            self.sess.close()

    def setDataset(self, dataset):
        """
        Function: Set dataset
        :param dataset:
        :return:
        """
        self.dataset = dataset

    def printLog(self, t_str):
        """
        Function: Display and log string
        :param t_str:
        :return:
        """
        # Add time stamp
        t_str = Utils.getTimeStamp()+': '+t_str
        # Print on screen
        print t_str
        # Write into file
        if not isinstance(self.log, int):
            self.log.write(t_str+'\r\n')
            self.log.flush()

    def buildGraph(self, name='SuperModel'):
        """
        Function: Build basic running graph
        Note: This function should be overridden by subclass
        :param name:
        :return:
        """

        with tf.variable_scope(name):

            a = tf.constant(value=[1, 2, 3], dtype=tf.float32)
            b = tf.constant(value=[5, 6, 7], dtype=tf.float32)

            c = a + b
        self.output = c
        raise NotImplementedError('Not implement')

    def buildLossGraph(self):
        """
        Function: Build a graph to describe loss function
        Note: This function should be overridden by subclass
        :return:
        """

        with tf.variable_scope('LossGraph'):
            self.loss = 0
        raise NotImplementedError('Not implement')

    def buildTrainingGraph(self, batchPerEpoch):
        """
        Function: Build a graph to describe training configurations
        :param batchPerEpoch: the number of batches for an epoch
        :return:
        """

        with tf.variable_scope('TrainingGraph'):
            # Compute gradient
            varList = tf.trainable_variables()
            grads = tf.gradients(self.loss, varList)
            grads, _ = tf.clip_by_global_norm(
                grads,
                self.config.maxGradNorm)

            # Set learning rate
            # startLearningRate*(learningStepFactor)^(global_step/batchPerEpoch)
            global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False)
            learningRate = tf.train.exponential_decay(
                self.config.startLearningRate,
                global_step,
                batchPerEpoch,
                self.config.learningStepFactor,
                staircase=True)
            self.learningRate = tf.maximum(
                learningRate,
                self.config.minLearningRate)

            # Apply optimizer
            opt = tf.train.AdamOptimizer(self.learningRate)
            self.trainOp = opt.apply_gradients(
                zip(grads, varList),
                global_step=global_step)

    def buildTaskGraph(self):
        """
        Function: Build the whole graph of the tensorflow model
        :return:
        """

        # Build graph
        self.buildGraph()
        self.buildLossGraph()
        # Check dataset
        if isinstance(self.dataset, int):
            raise ValueError('Empty dataset')
        if self.config.isTrain:
            # Build training graph
            self.config.minDispStep = \
                self.dataset.trainset.getSampleNum() / self.config.batchSize
            self.config.maxTrainTimes = \
                self.config.maxTrainEpoch * self.config.minDispStep
            self.buildTrainingGraph(self.config.minDispStep)
        # Build summary graph
        self.merge = tf.summary.merge_all()

    def createSession(self):
        """
        Function: Create a session in tensorflow
        :return:
        """
        # Check
        if not isinstance(self.sess, int):
            print Warning('Session has exist')
            return self.sess
        self.buildTaskGraph()
        # sessConf = tf.ConfigProto(
        #     # allow_soft_placement=True,
        #     log_device_placement=True)
        sessConf = tf.ConfigProto()
        if not self.config.isFullMemory:
            sessConf.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sessConf)
        self.sess.run(tf.global_variables_initializer())
        # Summary
        dir = os.path.join(self.config.modelDir, 'summary')
        if not os.path.isdir(dir):
            os.makedirs(dir)
        self.summary = tf.summary.FileWriter(dir, graph=self.sess.graph)
        # Return
        return self.sess

    def trainModel(self, modelDir='', sess=None):
        """
        Function: Train model
        :param modelDir:
        :param sess:
        :return:
        """
        print 'use to train model'
        raise NotImplementedError('Not implement')

    def testModel(self, modelDir='', sess=None):
        """
        Function: Test model
        :param modelDir:
        :param sess:
        :return:
        """
        if sess is None:
            sess = self.createSession()
        print 'use to test model'
        raise NotImplementedError('Not implement')

    def predict(self):
        """
        Function: Predict output
        :return:
        """
        print 'use to predict'
        raise NotImplementedError('Not implement')

    def saveTrainingCurve(self, scalarList, name):
        """
        Function: Save learning curve from scalar list
        :param scalarList:
        :param name:
        :return:
        """
        scalarArray = np.array(scalarList)
        step = np.arange(0, len(scalarList))
        filePath = os.path.join(self.config.modelDir, name+'.png')
        # Plot curve
        plt.clf()
        plt.title(name)
        plt.plot(step, scalarArray, '-ro')
        foo_fig = plt.gcf()  # 'get current figure
        foo_fig.savefig(filePath, format='png', dpi=128, bbox_inches='tight')
        # Utils.saveFig(step, scalarArray, filePath, name)


def main():
    print sys.argv

    model = SuperModel()
    model.trainModel()


if __name__ == '__main__':
    main()
