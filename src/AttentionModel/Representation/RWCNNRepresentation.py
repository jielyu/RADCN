# encoding:utf-8


# System libraries
# from __future__ import absolute_import
import sys
sys.path.append('..')
# 3rd-part libraries
import numpy as np
from scipy import misc
import tensorflow as tf

# Self-define libraries
from src.tools.Utils import Utils


# Define and implement a class
# to configure some parameters of class CNNRepresentation
class Config(object):

    def __init__(self,
                 inputShape=[48, 48, 3],
                 # pool3Shape=[1, 1, 1, 1],
                 numHiddenFc1=512,
                 isTrain=False,
                 keepProb=0.5,
                 isDropout=False):
        """
        Function: Initialize parameters for CNNRepresentation
        :param inputShape:  list, 3, default: [480, 640, 3]
        :param w1Shape:     list, 4, default: [5, 5, 3, 32]
        :param b1Shape:     list, 1, default: [32]
        :param pool1Shape:  list, 4, default: [1, 2, 2, 1]
        :param w2Shape:     list, 4, default, [5, 5, 32, 64]
        :param b2Shape:     list, 1, default, [64]
        :param pool2Shape:  list, 4, default: [1, 2, 2, 1]
        :param w3Shape:     list, 4, default: [5, 5, 64, 4]
        :param b3Shape:     list, 1, default: [4]
        :param pool3Shape:  list, 4, default: [1, 1, 1, 1]
        :param numHiddenFc1:    int, default: 512
        :param isTrain:     boolean, default: False
        :param keepProb:    float, default: 0.5
        """

        w1Shape = [5, 5, inputShape[-1], 64]
        b1Shape = [64]
        w2Shape = [5, 5, 64, 64]
        b2Shape = [64]
        w3Shape = [5, 5, 64, 64]
        b3Shape = [64]
        w4Shape = [5, 5, 64, 64]
        b4Shape = [64]
        w5Shape = [5, 5, 64, 64]
        b5Shape = [64]
        w6Shape = [5, 5, 64, 4]
        b6Shape = [4]

        # The shape of input images
        self.inputShape = inputShape

        # The shape of parameters on Conv1 layers
        self.w1Shape = w1Shape
        self.b1Shape = b1Shape
        pool1Shape = [1, 2, 2, 1]
        self.pool1Shape = pool1Shape

        # The shape of parameters on Conv2 layers
        self.w2Shape = w2Shape
        self.b2Shape = b2Shape
        pool2Shape = [1, 2, 2, 1]
        self.pool2Shape = pool2Shape

        # The shape of parameters on Conv3 layers
        self.w3Shape = w3Shape
        self.b3Shape = b3Shape
        pool3Shape = [1, 1, 1, 1]
        self.pool3Shape = pool3Shape

        # The shape of parameters on Conv4 layers
        self.w4Shape = w4Shape
        self.b4Shape = b4Shape
        pool4Shape = [1, 1, 1, 1]
        self.pool4Shape = pool4Shape

        # The shape of parameters on Conv5 layers
        self.w5Shape = w5Shape
        self.b5Shape = b5Shape
        pool5Shape = [1, 1, 1, 1]
        self.pool5Shape = pool5Shape

        # The shape of parameters on Conv6 layers
        self.w6Shape = w6Shape
        self.b6Shape = b6Shape
        pool6Shape = [1, 1, 1, 1]
        self.pool6Shape = pool6Shape

        # # The shape of parameters on Conv7 layers
        # self.w7Shape = w7Shape
        # self.b7Shape = b7Shape
        # pool7Shape = [1, 1, 1, 1]
        # self.pool7Shape = pool7Shape
        #
        # # The shape of parameters on Conv8 layers
        # self.w8Shape = w8Shape
        # self.b8Shape = b8Shape
        # pool8Shape = [1, 1, 1, 1]
        # self.pool8Shape = pool8Shape
        #
        # # The shape of parameters on Conv3 layers
        # self.w9Shape = w9Shape
        # self.b9Shape = b9Shape
        # pool9Shape = [1, 1, 1, 1]
        # self.pool8Shape = pool8Shape
        #
        # # The shape of parameters on Conv4 layers
        # self.w4Shape = w4Shape
        # self.b4Shape = b4Shape
        # pool4Shape = [1, 1, 1, 1]
        # self.pool4Shape = pool4Shape
        #
        # # The shape of parameters on Conv5 layers
        # self.w5Shape = w5Shape
        # self.b5Shape = b5Shape
        # pool5Shape = [1, 1, 1, 1]
        # self.pool5Shape = pool5Shape
        #
        # # The shape of parameters on Conv6 layers
        # self.w6Shape = w6Shape
        # self.b6Shape = b6Shape
        # pool6Shape = [1, 1, 1, 1]
        # self.pool6Shape = pool6Shape

        # The shape of Full connected layer
        numPixel = self.inputShape[0] * self.inputShape[1] / \
                   (self.pool1Shape[1] * self.pool1Shape[2] *
                    self.pool2Shape[1] * self.pool2Shape[2] *
                    self.pool3Shape[1] * self.pool3Shape[2] *
                    self.pool4Shape[1] * self.pool4Shape[2] *
                    self.pool5Shape[1] * self.pool5Shape[2] *
                    self.pool6Shape[1] * self.pool6Shape[2]) * \
                   self.w6Shape[3]
        self.w_fc1Shape = [numPixel, numHiddenFc1]
        self.b_fc1Shape = [numHiddenFc1]
        self.featureDim = numHiddenFc1

        # The flag whether enable taining configuration
        self.isTrain = isTrain
        # The dropout rate
        self.keepProb = keepProb
        self.isDropout = isDropout


# Define and implement a class to build a graph
# for extracting representations from images via CNN
class CNNRepresentation(object):

    def __init__(self, config=Config()):
        self.config = config

    def buildGraph(self, image, name='CNNRepresentation', reuse=False):
        """
        Function: Create a graph to extract representations by CNN
        :param image: 4D, [batchSize, height, width, channel]
        :return:
            fc1_norm: 2D, [batchSize, self.numHiddenFc1]
        """

        # Input images
        x = image
        with tf.variable_scope(name, reuse=reuse):
            nch = 16
            size = 3
            # Conv1
            shape = [size, size, self.config.inputShape[-1], nch]
            strides = 1
            self.w1, self.b1, self.conv1 = Utils.buildConvLayer(
                x=image,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv1')
            self.pool1 = self.conv1
            tf.summary.image(name='conv1',
                             tensor=tf.expand_dims(self.conv1[:, :, :, 0], axis=-1),
                             max_outputs=1)
            # Conv2
            shape = [size, size, nch, nch]
            self.w2, self.b2, self.conv2 = Utils.buildConvLayer(
                x=self.pool1,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv2')
            self.pool2 = self.conv2
            tf.summary.image(name='conv2',
                             tensor=tf.expand_dims(self.conv2[:, :, :, 0], axis=-1),
                             max_outputs=1)
            # Conv3
            shape = [size, size, nch, nch]
            self.w3, self.b3, self.conv3 = Utils.buildConvLayer(
                x=self.pool2,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv3')
            # print self.conv3
            self.pool3 = self.conv3 + self.conv1
            tf.summary.image(name='conv3',
                             tensor=tf.expand_dims(self.conv3[:, :, :, 0], axis=-1),
                             max_outputs=1)
            # Conv4
            shape = [size, size, nch, nch]
            self.w4, self.b4, self.conv4 = Utils.buildConvLayer(
                x=self.pool3,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv4')
            self.pool4 = self.conv4 + self.conv2
            # Conv5
            shape = [size, size, nch, nch]
            self.w5, self.b5, self.conv5 = Utils.buildConvLayer(
                x=self.pool4,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv5')
            self.pool5 = self.conv5 + self.conv3 + self.conv1
            # Conv6
            shape = [size, size, nch, nch]
            self.w6, self.b6, self.conv6 = Utils.buildConvLayer(
                x=self.pool5,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv6')
            # print self.conv6
            self.pool6 = self.conv6 + self.conv4 + self.conv2
            # Conv7
            shape = [size, size, nch, nch]
            self.w7, self.b7, self.conv7 = Utils.buildConvLayer(
                x=self.pool6,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv7')
            self.pool7 = self.conv7 + self.conv5 + self.conv3 + self.conv1
            # Conv8
            shape = [size, size, nch, nch]
            self.w8, self.b8, self.conv8 = Utils.buildConvLayer(
                x=self.pool7,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv8')
            self.pool8 = self.conv8 + self.conv6 + self.conv4 + self.conv2
            # Conv9
            shape = [size, size, nch, nch]
            self.w9, self.b9, self.conv9 = Utils.buildConvLayer(
                x=self.pool8,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv9')
            self.pool9 = self.conv9 + self.conv7 + self.conv5 + self.conv3 + self.conv1
            # Conv10
            shape = [size, size, nch, nch]
            self.w10, self.b10, self.conv10 = Utils.buildConvLayer(
                x=self.pool9,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv10')
            self.pool10 = self.conv10 + self.conv8 + self.conv6 + self.conv4 + self.conv2
            tf.summary.image(name='conv10',
                             tensor=tf.expand_dims(self.conv10[:, :, :, 0], axis=-1),
                             max_outputs=1)
            # Conv11
            shape = [size, size, nch, nch]
            pool_strd = 2
            self.w11, self.b11, self.conv11 = Utils.buildConvLayer(
                x=self.pool10,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv11')
            poolShape = [1, pool_strd, pool_strd, 1]
            self.pool11 = tf.nn.max_pool(
                value=self.conv11,
                ksize=poolShape,
                strides=poolShape,
                padding='SAME',
                name='pool11')
            # print self.conv11
            # Conv12
            och = 4
            shape = [size, size, nch, och]
            self.w12, self.b12, self.conv12 = Utils.buildConvLayer(
                x=self.pool11,
                w_shape=shape,
                strides=strides,
                reuse=reuse,
                name='Conv12')
            self.pool12 = tf.nn.max_pool(
                value=self.conv12,
                ksize=poolShape,
                strides=poolShape,
                padding='SAME',
                name='pool12')
            # print self.conv12

            # Fully-connected layers
            numPixel = self.config.inputShape[0]*self.config.inputShape[1]
            numPixel /= (pool_strd*pool_strd*pool_strd*pool_strd)
            numPixel *= och
            # print numPixel
            shape = [numPixel, self.config.featureDim]
            self.vec = tf.reshape(self.pool12,
                                  shape=[-1, shape[0]])
            self.w_fc1, self.b_fc1, self.fc1 = \
                Utils.buildFCLayer(self.vec, w_shape=shape, reuse=reuse, name='FC1')

            # Dropout
            if self.config.isTrain and self.config.isDropout:
                self.fc1 = tf.nn.dropout(
                    self.fc1,
                    keep_prob=self.config.keepProb,
                    name='fc1_dropout')
        # Return
        return self.fc1
            # self.w1 = Utils.getTFVariable(
            #     name='w1',
            #     shape=self.config.w1Shape)
            # self.b1 = Utils.getTFVariable(
            #     name='b1',
            #     shape=self.config.b1Shape)
            # self.conv1 = tf.nn.relu(
            #     tf.nn.conv2d(
            #         input=x,
            #         filter=self.w1,
            #         strides=[1, 1, 1, 1],
            #         padding='SAME')
            #     + self.b1,
            #     name='conv1')
        #     tf.summary.image(name='conv1',
        #                      tensor=tf.expand_dims(self.conv1[:, :, :, 0], axis=-1),
        #                      max_outputs=1)
        #     self.pool1 = tf.nn.max_pool(
        #         value=self.conv1,
        #         ksize=self.config.pool1Shape,
        #         strides=self.config.pool1Shape,
        #         padding='SAME',
        #         name='pool1')
        #
        #     # Conv2
        #     self.w2 = Utils.getTFVariable(
        #         name='w2',
        #         shape=self.config.w2Shape)
        #     self.b2 = Utils.getTFVariable(
        #         name='b2',
        #         shape=self.config.b2Shape)
        #     self.conv2 = tf.nn.relu(
        #         tf.nn.conv2d(
        #             input=self.pool1,
        #             filter=self.w2,
        #             strides=[1, 1, 1, 1],
        #             padding='SAME')
        #         + self.b2,
        #         name='conv2')
        #     tf.summary.image(name='conv2',
        #                      tensor=tf.expand_dims(self.conv2[:, :, :, 0], axis=-1),
        #                      max_outputs=1)
        #     self.pool2 = tf.nn.max_pool(
        #         value=self.conv2,
        #         ksize=self.config.pool2Shape,
        #         strides=self.config.pool2Shape,
        #         padding='SAME',
        #         name='pool2')
        #
        #     # Conv3
        #     self.w3 = Utils.getTFVariable(
        #         name='w3',
        #         shape=self.config.w3Shape)
        #     self.b3 = Utils.getTFVariable(
        #         name='b3',
        #         shape=self.config.b3Shape)
        #     self.conv3 = tf.nn.relu(
        #         tf.nn.conv2d(
        #             input=self.pool2,
        #             filter=self.w3,
        #             strides=[1, 1, 1, 1],
        #             padding='SAME')
        #         + self.b3,
        #         name='conv3')
        #     tf.summary.image(name='conv3',
        #                      tensor=tf.expand_dims(self.conv3[:, :, :, 0], axis=-1),
        #                      max_outputs=1)
        #     self.pool3 = self.conv3
        #     # self.pool3 = tf.nn.max_pool(
        #     #     value=self.conv3,
        #     #     ksize=self.config.pool3Shape,
        #     #     strides=self.config.pool3Shape,
        #     #     padding='SAME',
        #     #     name='pool3')
        #
        #     # Conv4
        #     self.w4 = Utils.getTFVariable(
        #         name='w4',
        #         shape=self.config.w4Shape)
        #     self.b4 = Utils.getTFVariable(
        #         name='b4',
        #         shape=self.config.b4Shape)
        #     self.conv4 = tf.nn.relu(
        #         tf.nn.conv2d(
        #             input=self.pool3,
        #             filter=self.w4,
        #             strides=[1, 1, 1, 1],
        #             padding='SAME')
        #         + self.b4,
        #         name='conv4')
        #     self.pool4 = self.conv4
        #     # Conv5
        #     self.w5 = Utils.getTFVariable(
        #         name='w5',
        #         shape=self.config.w5Shape)
        #     self.b5 = Utils.getTFVariable(
        #         name='b5',
        #         shape=self.config.b5Shape)
        #     self.conv5 = tf.nn.relu(
        #         tf.nn.conv2d(
        #             input=self.pool4,
        #             filter=self.w5,
        #             strides=[1, 1, 1, 1],
        #             padding='SAME')
        #         + self.b5,
        #         name='conv5')
        #     self.pool5 = self.conv5
        #
        #     # Conv6
        #     self.w6 = Utils.getTFVariable(
        #         name='w6',
        #         shape=self.config.w6Shape)
        #     self.b6 = Utils.getTFVariable(
        #         name='b6',
        #         shape=self.config.b6Shape)
        #     self.conv6 = tf.nn.relu(
        #         tf.nn.conv2d(
        #             input=self.pool5,
        #             filter=self.w6,
        #             strides=[1, 1, 1, 1],
        #             padding='SAME')
        #         + self.b6,
        #         name='conv6')
        #     self.pool6 = self.conv6
        #     # Reshape
        #     # print self.config.w_fc1Shape
        #     self.vec = tf.reshape(self.pool6,
        #                           shape=[-1, self.config.w_fc1Shape[0]])
        #
        #     # Full connected
        #     self.w_fc1 = Utils.getTFVariable(
        #         name='w_fc1',
        #         shape=self.config.w_fc1Shape)
        #     self.b_fc1 = Utils.getTFVariable(
        #         name='b_fc1',
        #         shape=self.config.b_fc1Shape)
        #     self.fc1 = tf.nn.relu(
        #         tf.matmul(self.vec, self.w_fc1)
        #         + self.b_fc1)
        #
        #     # Dropout
        #     if self.config.isTrain and self.config.isDropout:
        #         self.fc1_drop = tf.nn.dropout(
        #             self.fc1,
        #             keep_prob=self.config.keepProb,
        #             name='fc1_dropout')
        #     else:
        #         self.fc1_drop = self.fc1
        #
        #     # Normalization
        #     # self.fc1_norm = tf.nn.l2_normalize(self.fc1_drop,
        #     #                                    dim=1,
        #     #                                    name='fc_norm')l
        #     self.fc1_norm = self.fc1_drop
        #
        #     # Softmax
        #     self.fc1_sm = tf.nn.softmax(self.fc1_drop, name='fc1_sm')
        #
        # # Return
        # return self.fc1_norm


# The main function of the demo
def main():
    print sys.argv

    images = tf.constant(value=128.0, dtype=tf.float32, shape=[64, 48, 48, 3])
    convNet = CNNRepresentation()
    feat = convNet.buildGraph(image=images)
    print feat


# The entry of the demo
if __name__ == '__main__':
    main()
