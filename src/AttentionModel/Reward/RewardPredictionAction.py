# encoding: utf8

import tensorflow as tf

from src.tools.Utils import Utils


class Config:

    def __init__(self,
                 isTrain=False,
                 featureDim=256,
                 keepProb=1.0,
                 isDropout=False):
        # The flag of training state
        self.isTrain = isTrain
        # The dimension of input features
        self.featureDim = featureDim
        self.rewardDim = 1
        # The probability of keeping effectiveness in dropout
        self.keepProb = keepProb
        self.isDropout = isDropout


class RewardPredictionAction:

    def __init__(self, config=Config()):
        self.config = config

        # self.w_fc1 = Utils.getTFVariable(
        #     name='w_fc1',
        #     shape=[self.config.featureDim, self.config.rewardDim])
        # self.b_fc1 = Utils.getTFVariable(
        #     name='b_fc1',
        #     shape=[self.config.rewardDim])
        # self.fc1 = 0
        #
        # self.fc1_dropout = 0
        # self.fc1_clip = 0

    def buildGraph(self, feat, name='RewardPredictionAction', reuse=False):
        # Create name scope
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # Full connect
            self.w_fc1 = Utils.getTFVariable(
                name='w_fc1',
                shape=[self.config.featureDim, self.config.rewardDim])
            self.b_fc1 = Utils.getTFVariable(
                name='b_fc1',
                shape=[self.config.rewardDim])
            self.fc1 = tf.matmul(feat, self.w_fc1) + self.b_fc1
            # Dropout
            if self.config.isTrain and self.config.isDropout:
                self.fc1_dropout = tf.nn.dropout(
                    x=self.fc1,
                    keep_prob=self.config.keepProb)
            else:
                self.fc1_dropout = self.fc1

            self.fc1_clip = tf.clip_by_value(self.fc1_dropout, 0, 1.0, 'clip')

        preReward = self.fc1_clip
        # Return
        return preReward
