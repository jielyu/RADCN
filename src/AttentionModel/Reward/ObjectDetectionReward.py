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
                 objectDim=5,
                 isTrain=False,
                 keepProb=0.5,
                 isDropout=False):
        """
        Function: INitialize the parameters for ObjectDetectionReward
        :param featureDim: int, default: 512
        :param numHiddenUnits:  int, default: 128
        :param objectDim:   int, default: 5
        :param isTrain:     boolean, default: False
        :param keepProb:    float, default: 0.5
        """
        # The dimension of features
        self.featureDim = featureDim
        # The number of hidden units on full connected layers
        self.numHiddenUnits = numHiddenUnits
        # The dimension of a object description
        self.objectDim = objectDim # x,y,w,h,score
        # The flag whether enable training configuration
        self.isTrain = isTrain
        # The dropout rate setting
        self.keepProb = keepProb
        self.isDropout = isDropout


# Define and implement a class
# to build reward network for object detection task
class ObjectDetectionReward(object):

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
        # self.w_fc2 = Utils.getTFVariable(name='w_fc2',
        #                                  shape=[self.config.numHiddenUnits,
        #                                         self.config.objectDim])
        #
        # self.b_fc2 = Utils.getTFVariable(name='b_fc2',
        #                                  shape=[self.config.objectDim])
        # self.fc2 = 0
        #
        # if self.config.objectDim != 5:
        #     raise ValueError('Not yx-wh-score object detection')
        # self.w_fc_yx = Utils.getTFVariable(name='w_fc_yx',
        #                                    shape=[self.config.numHiddenUnits,
        #                                           2])
        # self.b_fc_yx = Utils.getTFVariable(name='b_fc_yx',
        #                                    shape=[2])
        # self.w_fc_hw = Utils.getTFVariable(name='w_fc_hw',
        #                                    shape=[self.config.numHiddenUnits,
        #                                           2])
        # self.b_fc_hw = Utils.getTFVariable(name='b_fc_hw',
        #                                    shape=[2])
        # self.w_fc_score = \
        #     Utils.getTFVariable(name='w_fc_score',
        #                         shape=[self.config.numHiddenUnits,
        #                                1])
        # self.b_fc_score = Utils.getTFVariable(name='b_fc_score',
        #                                       shape=[1])
        #
        # # Limit range
        # self.yx = 0
        # self.hw = 0
        # self.score = 0

    def buildGraph(self, feat, name='ObjectDetetcionReward', reuse=False):
        """
        Function: Create a graph to predict reward for object detection task
        :param feat: 2D, [batchSize, featureDim]
        :return:
            yx,     the coordinates of the center point
            hw,     the height and width of the rectangle
            score,  the score of the rectangle
        """
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
            # self.fc1_norm = tf.nn.l2_normalize(self.fc1_dropout,
            #                                    dim=1,
            #                                    name='fc_norm')
            self.fc1_norm = self.fc1_dropout

            # Prediction
            # self.w_fc2 = Utils.getTFVariable(name='w_fc2',
            #                                  shape=[self.config.numHiddenUnits,
            #                                         self.config.objectDim])
            #
            # self.b_fc2 = Utils.getTFVariable(name='b_fc2',
            #                                  shape=[self.config.objectDim])
            # self.fc2 = tf.nn.sigmoid(
            #     x=tf.matmul(self.fc1_norm, self.w_fc2) + self.b_fc2,
            #     name='fc2')
            # self.fc2 = tf.matmul(self.fc1_norm, self.w_fc2) + self.b_fc2
            #
            # self.yx = tf.clip_by_value(self.fc2[:, 0:2], -1.0, 1.0)  # [-1,1]
            # self.hw = tf.clip_by_value(self.fc2[:, 2:4], 0.0, 2.0)  # [0,2]
            # self.score = tf.clip_by_value(self.fc2[:, 4], 0.0, 1.0)  #
            # Limit range
            # self.yx = 2 * self.fc2[:, 0:2] - 1  # [-1,1]
            # self.hw = 2 * self.fc2[:, 2:4]  # [0,2]
            # self.score = self.fc2[:, 4]  # [0,1]

            if self.config.objectDim != 5:
                raise ValueError('Not yx-wh-score object detection')
            self.w_fc_yx = Utils.getTFVariable(name='w_fc_yx',
                                               shape=[self.config.numHiddenUnits,
                                                      2])
            self.b_fc_yx = Utils.getTFVariable(name='b_fc_yx',
                                               shape=[2])
            self.yx = tf.matmul(self.fc1_norm, self.w_fc_yx) + self.b_fc_yx

            self.w_fc_hw = Utils.getTFVariable(name='w_fc_hw',
                                               shape=[self.config.numHiddenUnits,
                                                      2])
            self.b_fc_hw = Utils.getTFVariable(name='b_fc_hw',
                                               shape=[2])
            self.hw = \
                tf.matmul(self.fc1_norm, self.w_fc_hw) + self.b_fc_hw

            self.w_fc_score = \
                Utils.getTFVariable(name='w_fc_score',
                                    shape=[self.config.numHiddenUnits,
                                           1])
            self.b_fc_score = Utils.getTFVariable(name='b_fc_score',
                                                  shape=[1])
            self.score = \
                tf.matmul(self.fc1_norm, self.w_fc_score) + self.b_fc_score

            self.yx = tf.clip_by_value(self.yx, -1.0, 1.0)  # [-1,1]
            self.hw = tf.clip_by_value(self.hw, 0.0, 2.0)  # [0,2]
            self.score = tf.clip_by_value(self.score, 0.0, 1.0)  # [0, 1]
            # Return
            return self.yx, self.hw, self.score


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
    objectDetector = ObjectDetectionReward()
    # Build graph
    obj = objectDetector.buildGraph(x_feat)
    # Initial op
    initVar = tf.initialize_all_variables()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(initVar)
        # Run graph
        objs = sess.run(obj)
        # Print
        print objs


# The entry of the demo
if __name__ == '__main__':
    main()

