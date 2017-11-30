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
                 inputShape=[480, 640, 3],
                 w1Shape=[5, 5, 3, 32],
                 b1Shape=[32],
                 pool1Shape=[1, 2, 2, 1],
                 w2Shape=[5, 5, 32, 64],
                 b2Shape=[64],
                 pool2Shape=[1, 2, 2, 1],
                 w3Shape=[5, 5, 64, 4],
                 b3Shape=[4],
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

        # The shape of input images
        self.inputShape = inputShape

        # The shape of parameters on Conv1 layers
        self.w1Shape = w1Shape
        self.b1Shape = b1Shape
        self.pool1Shape = pool1Shape

        # The shape of parameters on Conv2 layers
        self.w2Shape = w2Shape
        self.b2Shape = b2Shape
        self.pool2Shape = pool2Shape

        # The shape of parameters on Conv3 layers
        self.w3Shape = w3Shape
        self.b3Shape = b3Shape
        pool3Shape = [1, 1, 1, 1]
        self.pool3Shape = pool3Shape

        # The shape of Full connected layer
        numPixel = self.inputShape[0] * self.inputShape[1] / \
                   (self.pool1Shape[1] * self.pool1Shape[2]*
                    self.pool2Shape[1] * self.pool2Shape[2] *
                    self.pool3Shape[1] * self.pool3Shape[2]) * \
                   self.w3Shape[3]
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

        # # Conv2
        # self.w1 = Utils.getTFVariable(
        #     name='w1',
        #     shape=self.config.w1Shape)
        # self.b1 = Utils.getTFVariable(
        #     name='b1',
        #     shape=self.config.b1Shape)
        # self.conv1 = 0
        # self.pool1 = 0
        #
        # # Conv2
        # self.w2 = Utils.getTFVariable(
        #     name='w2',
        #     shape=self.config.w2Shape)
        # self.b2 = Utils.getTFVariable(
        #     name='b2',
        #     shape=self.config.b2Shape)
        # self.conv2 = 0
        # self.pool2 = 0
        #
        # # Conv3
        # self.w3 = Utils.getTFVariable(
        #     name='w3',
        #     shape=self.config.w3Shape)
        # self.b3 = Utils.getTFVariable(
        #     name='b3',
        #     shape=self.config.b3Shape)
        # self.conv3 = 0
        # self.pool3 = 0
        #
        # # Reshape
        # self.vec = 0
        #
        # # Full connected
        # self.w_fc1 = Utils.getTFVariable(
        #     name='w_fc1',
        #     shape=self.config.w_fc1Shape)
        # self.b_fc1 = Utils.getTFVariable(
        #     name='b_fc1',
        #     shape=self.config.b_fc1Shape)
        # self.fc1 = 0
        #
        # # Dropout
        # self.fc1_drop = 0
        #
        # # Normalization
        # self.fc1_norm = 0
        #
        # # Softmax
        # self.fc1_sm = 0

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

            # Conv1
            self.w1 = Utils.getTFVariable(
                name='w1',
                shape=self.config.w1Shape)
            self.b1 = Utils.getTFVariable(
                name='b1',
                shape=self.config.b1Shape)
            self.conv1 = tf.nn.relu(
                tf.nn.conv2d(
                    input=x,
                    filter=self.w1,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
                + self.b1,
                name='conv1')
            tf.summary.image(name='conv1',
                             tensor=tf.expand_dims(self.conv1[:, :, :, 0], axis=-1),
                             max_outputs=1)
            self.pool1 = tf.nn.max_pool(
                value=self.conv1,
                ksize=self.config.pool1Shape,
                strides=self.config.pool1Shape,
                padding='SAME',
                name='pool1')

            # Conv2
            self.w2 = Utils.getTFVariable(
                name='w2',
                shape=self.config.w2Shape)
            self.b2 = Utils.getTFVariable(
                name='b2',
                shape=self.config.b2Shape)
            self.conv2 = tf.nn.relu(
                tf.nn.conv2d(
                    input=self.pool1,
                    filter=self.w2,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
                + self.b2,
                name='conv2')
            tf.summary.image(name='conv2',
                             tensor=tf.expand_dims(self.conv2[:, :, :, 0], axis=-1),
                             max_outputs=1)
            self.pool2 = tf.nn.max_pool(
                value=self.conv2,
                ksize=self.config.pool2Shape,
                strides=self.config.pool2Shape,
                padding='SAME',
                name='pool2')

            # Conv3
            self.w3 = Utils.getTFVariable(
                name='w3',
                shape=self.config.w3Shape)
            self.b3 = Utils.getTFVariable(
                name='b3',
                shape=self.config.b3Shape)
            self.conv3 = tf.nn.relu(
                tf.nn.conv2d(
                    input=self.pool2,
                    filter=self.w3,
                    strides=[1, 1, 1, 1],
                    padding='SAME')
                + self.b3,
                name='conv3')
            tf.summary.image(name='conv3',
                             tensor=tf.expand_dims(self.conv3[:, :, :, 0], axis=-1),
                             max_outputs=1)
            # self.pool3 = tf.nn.max_pool(
            #     value=self.conv3,
            #     ksize=self.config.pool3Shape,
            #     strides=self.config.pool3Shape,
            #     padding='SAME',
            #     name='pool3')

            # Reshape
            # print self.config.w_fc1Shape
            self.vec = tf.reshape(self.conv3,
                                  shape=[-1, self.config.w_fc1Shape[0]])

            # Full connected
            self.w_fc1 = Utils.getTFVariable(
                name='w_fc1',
                shape=self.config.w_fc1Shape)
            self.b_fc1 = Utils.getTFVariable(
                name='b_fc1',
                shape=self.config.b_fc1Shape)
            self.fc1 = tf.nn.relu(
                tf.matmul(self.vec, self.w_fc1)
                + self.b_fc1)

            # Dropout
            if self.config.isTrain and self.config.isDropout:
                self.fc1_drop = tf.nn.dropout(
                    self.fc1,
                    keep_prob=self.config.keepProb,
                    name='fc1_dropout')
            else:
                self.fc1_drop = self.fc1

            # Normalization
            # self.fc1_norm = tf.nn.l2_normalize(self.fc1_drop,
            #                                    dim=1,
            #                                    name='fc_norm')l
            self.fc1_norm = self.fc1_drop

            # Softmax
            self.fc1_sm = tf.nn.softmax(self.fc1_drop, name='fc1_sm')

        # Return
        return self.fc1_norm


# The main function of the demo
def main():
    print sys.argv

    # Read data
    # Images
    imgPath = \
        '/home/jielyu/Workspace/Python/AttentionModel/data/' \
        'test/v_0093_img_000011.jpg'
    img = misc.imread(imgPath)
    imgs = np.zeros(shape=[2, 480, 640, 3], dtype=np.float32)
    imgs[0, :] = img
    imgs[1, :] = img
    imgs = imgs / 255 - 0.5

    # Create input placeholder
    x_shape = [None, 480, 640, 3]
    x_place = tf.placeholder(dtype=tf.float32, shape=x_shape)

    # Build Graph
    cnnRep = CNNRepresentation()
    feat = cnnRep.buildGraph(x_place)
    initOp = tf.initialize_all_variables()

    merged_summaries = tf.merge_all_summaries()
    # Create session
    with tf.Session() as sess:
        # Initialize variables
        sess.run(initOp)
        # Run the graph
        feed_dict = {x_place: imgs}
        output = sess.run(feat, feed_dict=feed_dict)
        # Output
        print output.shape
        print output.dtype
        print output
        print output[0, 10:20]

    writer = tf.train.SummaryWriter(
        '/home/jielyu/Workspace/Python/AttentionModel/tf_graph/name_scope_2',
        graph=tf.get_default_graph())
    writer.close()


# The entry of the demo
if __name__ == '__main__':
    main()
