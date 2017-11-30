# encoding:utf-8


# System libraries
import sys
sys.path.append('..')
# 3rd-part libraries
import numpy as np
import tensorflow as tf

# Self-define libraries
from src.tools.Utils import Utils


# Define and implement a class
class Config(object):

    def __init__(self,
                 featureDim=512,
                 numCategory=10,
                 isTrain=False,
                 keepProb=0.5,
                 isDropout=False):

        # The dimension of input features
        self.featureDim = featureDim
        # The number of categories
        self.numCategory = numCategory
        # The flag of training state
        self.isTrain = isTrain
        # The probability of keeping effectiveness in dropout
        self.keepProb = keepProb
        self.isDropout = isDropout


# Define and implement a class
# to classify samples into several categories
class ClassificationAction(object):

    def __init__(self, config=Config()):

        # The configuration of the ClassificationAction object
        self.config = config

        # self.w_fc2 = Utils.getTFVariable(
        #     name='w_fc1',
        #     shape=[self.config.featureDim, self.config.featureDim])
        # self.b_fc2 = Utils.getTFVariable(
        #     name='b_fc1',
        #     shape=[self.config.featureDim])
        # self.fc2 = 0
        #
        # self.w_fc1 = Utils.getTFVariable(
        #     name='w_fc1',
        #     shape=[self.config.featureDim, self.config.numCategory])
        # self.b_fc1 = Utils.getTFVariable(
        #     name='b_fc1',
        #     shape=[self.config.numCategory])
        # self.fc1 = 0
        #
        # self.fc1_dropout = 0
        # self.softmax = 0

    def buildGraph(self, feat, name='ClassificationAction', reuse=False):
        """
        Function: Build a graph to classify samples into several categories
        :param feat: float, 2D, [None, config.numFeature]
        :return:
                the probabilities of all categories
                float(0,1), 2D, [None, config.categories]
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # Full connect
            self.w_fc2 = Utils.getTFVariable(
                name='w_fc2',
                shape=[self.config.featureDim, self.config.featureDim])
            self.b_fc2 = Utils.getTFVariable(
                name='b_fc2',
                shape=[self.config.featureDim])
            self.fc2 = tf.nn.relu(
                features=tf.matmul(feat, self.w_fc2) + self.b_fc2,
                name='fc2')
            if self.config.isTrain and self.config.isDropout:
                self.fc2 = tf.nn.dropout(
                    x=self.fc2,
                    keep_prob=self.config.keepProb)
            # self.fc1 = tf.matmul(feat, self.w_fc1) + self.b_fc1
            self.w_fc1 = Utils.getTFVariable(
                name='w_fc1',
                shape=[self.config.featureDim, self.config.numCategory])
            self.b_fc1 = Utils.getTFVariable(
                name='b_fc1',
                shape=[self.config.numCategory])
            self.fc1 = tf.nn.relu(
                features=tf.matmul(self.fc2, self.w_fc1) + self.b_fc1,
                name='fc1')
            # if self.config.isTrain and self.config.isDropout:
            #     self.fc1_dropout = tf.nn.dropout(
            #         x=self.fc1,
            #         keep_prob=self.config.keepProb)
            # else:
            #     self.fc1_dropout = self.fc1
            self.fc1_dropout = self.fc1

            # Classify via softmax
            self.softmax = tf.nn.softmax(self.fc1_dropout, name='softmax')

            # Return the probabilities of all categories
            return self.softmax


# The main function of the demo
def main():
    print sys.argv

    # Input feature
    x_feat = tf.truncated_normal(
        shape=[3, 512],
        mean=0,
        stddev=0.1,
        dtype=tf.float32)
    # Create object
    classificationAction = ClassificationAction()
    # Build graph
    prob = classificationAction.buildGraph(x_feat)
    # Initial op
    initVar = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(initVar)
        # Run graph
        probs = sess.run(prob)
        # Print
        print probs

        print np.sum(probs, 1)


# The entry of the demo
if __name__ == '__main__':
    main()
