# encoding:utf-8


# System libraries
import sys
sys.path.append('..')
# 3rd-part libraries
import tensorflow as tf

# Self-define libraries
from src.tools.Utils import Utils


# Define and implement a class
# to configure some parameters of class FocusedPointPrediction
class Config(object):

    def __init__(self,
                 featureDim=512,
                 numHiddenUnits=128,
                 coordinateDim=2,
                 isTrain=False,
                 keepProb=0.5,
                 isDropout=False):
        """
        Function: Initialize parameters for FocusedPointPrediction
        :param featureDim:      int, default: 512
        :param numHiddenUnits:  int, default: 128
        :param coordinateDim:   int, default: 2
        :param isTrain:         boolean, default: False
        :param keepProb:        float, default: 0.5
        """
        # The dimension of features
        self.featureDim = featureDim
        # The number of hidden units
        self.numHiddenUnits = numHiddenUnits
        # The dimension of focused points
        self.coordinateDim = coordinateDim
        # The flag whether enable training configuration
        self.isTrain = isTrain
        # The dropout rate
        self.keepProb = keepProb
        self.isDropout = isDropout


# Define and implement a class to build a graph
# for predicting the focused point at the next step
class FocusedPointPrediction(object):

    def __init__(self, config=Config()):
        self.config = config

        # # Full connect
        # self.w_fc1 = Utils.getTFVariable(
        #     name='w_fc1',
        #     shape=[self.config.featureDim, self.config.numHiddenUnits])
        # self.b_fc1 = Utils.getTFVariable(
        #     name='b_fc1',
        #     shape=[self.config.numHiddenUnits])
        # self.fc1 = 0
        # self.fc1_dropout = 0
        # self.fc1_norm = 0
        #
        # # Prediction
        # self.w_fc2 = Utils.getTFVariable(
        #     name='w_fc2',
        #     shape=[self.config.numHiddenUnits, self.config.coordinateDim])
        # self.b_fc2 = Utils.getTFVariable(
        #     name='b_fc2',
        #     shape=[self.config.coordinateDim])
        # self.fc2 = 0
        #
        # self.fc2_output = 0

    def buildGraph(self, feat, name='FocusedPointPrediction', reuse=False):
        """
        Function: Create a graph to predict the next focused points
        :param feat: 2D, [batchSize, featureDim]
        :return:
            fc2_output: 2D, [batchSize, 2], range(-1,1), float
        """

        # Focused points
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            # Full connect
            self.w_fc1 = Utils.getTFVariable(
                name='w_fc1',
                shape=[self.config.featureDim, self.config.numHiddenUnits])
            self.b_fc1 = Utils.getTFVariable(
                name='b_fc1',
                shape=[self.config.numHiddenUnits])
            self.fc1 = tf.nn.relu(
                features=tf.matmul(feat, self.w_fc1) + self.b_fc1,
                name='fc1')
            if self.config.isTrain and self.config.isDropout:
                self.fc1_dropout = tf.nn.dropout(
                    x=self.fc1,
                    keep_prob=self.config.keepProb)
            else:
                self.fc1_dropout = self.fc1

            # Normalization
            self.fc1_norm = tf.nn.l2_normalize(self.fc1_dropout,
                                               dim=1,
                                               name='fc_norm')

            # Prediction
            self.w_fc2 = Utils.getTFVariable(
                name='w_fc2',
                shape=[self.config.numHiddenUnits, self.config.coordinateDim])
            self.b_fc2 = Utils.getTFVariable(
                name='b_fc2',
                shape=[self.config.coordinateDim])
            self.fc2 = tf.nn.sigmoid(
                x=tf.matmul(self.fc1_norm, self.w_fc2) + self.b_fc2,
                name='fc2')

            # Limit [-1,1]
            self.fc2_output = self.fc2*2 - 1
            # Stop gradient accumulation
            # self.fc2_output = tf.stop_gradient(self.fc2_output)
            # Return the predicted focused point
            return self.fc2_output


# The main function of the demo
def main():
    print sys.argv

    # Input feature
    x_feat = tf.truncated_normal(
        shape=[3,512],
        mean=0,
        stddev=0.1,
        dtype=tf.float32)
    # Create object
    focusedPointPredictor = FocusedPointPrediction()
    # Build graph
    pt = focusedPointPredictor.buildGraph(x_feat)
    # Initial op
    initVar = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(initVar)
        # Run graph
        point = sess.run(pt)
        # Print
        print point


# The entry of the demo
if __name__ == '__main__':
    main()
