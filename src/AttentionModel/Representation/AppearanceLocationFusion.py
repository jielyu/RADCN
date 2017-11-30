# encoding:utf-8


# System libraries
# from __future__ import absolute_import
import sys
sys.path.append('..')
# 3rd-part libraries
import tensorflow as tf

# Self-define libraries
from src.tools.Utils import Utils


# The implementation of a class
# to configure the parameters of AppearanceLocationFusion object
class Config(object):

    def __init__(self,
                 visualFeatDim=256,
                 locationFeatDim=2,
                 featureDim=256,
                 method='plus'):
        self.visualFeatDim = visualFeatDim
        self.locationFeatDim = locationFeatDim
        self.featureDim = featureDim
        self.method = method


# The implementation of a class
# to achieve the fusion representation of appearance and location
class AppearanceLocationFusion(object):

    def __init__(self, config=Config()):
        self.config = config

        # shape = [config.visualFeatDim, config.featureDim]
        # self.w_fc1 = Utils.getTFVariable(name='w_fc1',
        #                                  shape=shape)
        # self.b_fc1 = Utils.getTFVariable(name='b_fc1',
        #                                  shape=[config.featureDim])
        # self.fc1 = 0
        #
        # shape = [config.locationFeatDim, config.featureDim]
        # self.w_fc2 = Utils.getTFVariable(name='w_fc2',
        #                                  shape=shape)
        # self.b_fc2 = Utils.getTFVariable(name='b_fc2',
        #                                  shape=[config.featureDim])
        # self.fc2 = 0
        #
        # if config.method == 'concat':
        #     shape = [config.featureDim*2,
        #              config.featureDim]
        #     self.w_output = Utils.getTFVariable(name='w_output',
        #                                         shape=shape)
        #     self.b_output = Utils.getTFVariable(name='b_output',
        #                                         shape=[config.featureDim])
        #
        # self.outputFeat = 0

    def buildGraph(self, visualFeat, locationFeat, name='AppearanceLocationFusion', reuse=False):

        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            # Map visual feature into specific dimension
            shape = [self.config.visualFeatDim, self.config.featureDim]
            self.w_fc1 = Utils.getTFVariable(name='w_fc1',
                                             shape=shape)
            self.b_fc1 = Utils.getTFVariable(name='b_fc1',
                                             shape=[self.config.featureDim])
            self.fc1 = tf.nn.relu(
                features=tf.matmul(visualFeat, self.w_fc1)+self.b_fc1,
                name='fc1')
            # Map location feature into specific dimension

            shape = [self.config.locationFeatDim, self.config.featureDim]
            self.w_fc2 = Utils.getTFVariable(name='w_fc2',
                                             shape=shape)
            self.b_fc2 = Utils.getTFVariable(name='b_fc2',
                                             shape=[self.config.featureDim])
            self.fc2 = tf.nn.relu(
                features=tf.matmul(locationFeat, self.w_fc2)+self.b_fc2,
                name='fc2')
            # Fusion visual and location features
            if self.config.method == 'plus':
                self.outputFeat = self.fc1 + self.fc2
            elif self.config.method == 'product':
                self.outputFeat = self.fc1 * self.fc2
            elif self.config.method == 'concat':
                shape = [self.config.featureDim * 2,
                         self.config.featureDim]
                self.w_output = Utils.getTFVariable(name='w_output',
                                                    shape=shape)
                self.b_output = Utils.getTFVariable(name='b_output',
                                                    shape=[self.config.featureDim])
                feat = tf.concat(concat_dim=1, values=[self.fc1, self.fc2])
                self.outputFeat = tf.nn.relu(
                    features=tf.matmul(feat, self.w_output) + self.b_output,
                    name='output')
            elif self.config.method == 'app_only':
                self.outputFeat = self.fc1
            else:
                raise ValueError('Not specific method')
        # Return
        return self.outputFeat


# The main function of the demo
def main():
    print sys.argv


# The entry of the demo
if __name__ == '__main__':
    main()
