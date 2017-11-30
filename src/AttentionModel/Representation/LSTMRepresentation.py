# encoding:utf-8


# System libraries
import os
import sys
sys.path.append('..')
# 3rd-part libraries
import tensorflow as tf

# Self-define libraries
from src.tools.Utils import Utils


# Define and implement a class
# to configure some parameters of class CNNRepresentation
class Config(object):

    def __init__(self,
                 numHiddenUnits=512,
                 visualFeatDim=512,
                 lstmLayers=3,
                 isTrain=False,
                 fcKeepProb=0.5,
                 lstmKeepProb=0.5,
                 isDropout=False):
        """
        Function: Initialize the parameters for setting LSTMRepresentation
        :param numHiddenUnits:  int, default:512
        :param visualFeatDim:   int, default:512
        :param lstmLayers:      int, default:3
        :param isTrain:         boolean, default:False
        :param fcKeepProb:      float, default:0.5
        :param lstmKeepProb:    float, default:0.5
        """
        # The number of hidden units of LSTM blocks
        self.numHiddenUnits = numHiddenUnits
        # The dimension of visual features
        self.visualFeatDim = visualFeatDim
        # The number of LSTM layers
        self.lstmLayers = lstmLayers
        # The flag whether enable training configuration
        self.isTrain = isTrain
        # The dropout rate of full connected layer
        self.fcKeepProb = fcKeepProb
        # The dropout rate of LSTM Net
        self.lstmKeepProb = lstmKeepProb
        self.isDropout = isDropout
        # self.focusedCoordinateNum = focusedCoordinateNum
        # The batch size
        # self.batchSize = batchSize


# Define and implement a class to build a graph
# for combining the states at last step and visual representation
class LSTMRepresentation(object):

    def __init__(self, config=Config()):
        # The configuration of this object
        self.config = config
        # The flag indicating initial state
        self.isInitial = True
        # The states of LSTM
        self.states = 0

        # Fully connected layer
        # visualDim = self.config.numHiddenUnits
        # self.w_fc1 = Utils.getTFVariable(
        #     name='w_fc1',
        #     shape=[self.config.visualFeatDim, visualDim])
        # self.b_fc1 = Utils.getTFVariable(
        #     name='b_fc1',
        #     shape=[visualDim])
        # self.fc1 = 0

        # # LSTM dropout
        # self.lstm_cell = tf.contrib.rnn.LSTMCell(
        #     num_units=self.config.numHiddenUnits,
        #     forget_bias=1.0,
        #     state_is_tuple=True)
        # if self.config.isTrain:
        #     self.lstm_cell = tf.contrib.rnn.DropoutWrapper(
        #         cell=self.lstm_cell,
        #         output_keep_prob=self.config.lstmKeepProb)
        #
        # # Multi-layer LSTM
        # self.cell = tf.contrib.rnn.MultiRNNCell(
        #     cells=[self.lstm_cell for _ in range(self.config.lstmLayers)],
        #     state_is_tuple=True)
        # if self.config.isTrain:
        #     self.cell = tf.contrib.rnn.MultiRNNCell(
        #         cells=[tf.contrib.rnn.DropoutWrapper(
        #             cell=tf.contrib.rnn.LSTMCell(
        #                 num_units=self.config.numHiddenUnits,
        #                 forget_bias=1.0,
        #                 state_is_tuple=True),
        #             output_keep_prob=self.config.lstmKeepProb)
        #                for _ in range(self.config.lstmLayers)],
        #         state_is_tuple=True)
        # else:
        #     self.cell = tf.contrib.rnn.MultiRNNCell(
        #         cells=[tf.contrib.rnn.LSTMCell(
        #             num_units=self.config.numHiddenUnits,
        #             forget_bias=1.0,
        #             state_is_tuple=True)
        #                for _ in range(self.config.lstmLayers)],
        #         state_is_tuple=True)
        #
        # self.outputs = 0

    def buildGraph(self, visualFeat, name='LSTMRepresentation', reuse=False):
        """
        Function: build a graph to combine visual
                  features with the states at last step
        :param visualFeat: 2D, [batchSize, featureDim]
        :return:
                2D, [batchSize, featureDim]
        """

        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            # Construct cell
            if not reuse:
                if self.config.isTrain:
                    self.cell = tf.contrib.rnn.MultiRNNCell(
                        cells=[tf.contrib.rnn.DropoutWrapper(
                            cell=tf.contrib.rnn.LSTMCell(
                                num_units=self.config.numHiddenUnits,
                                forget_bias=1.0,
                                state_is_tuple=True),
                            output_keep_prob=self.config.lstmKeepProb)
                               for _ in range(self.config.lstmLayers)],
                        state_is_tuple=True)
                else:
                    self.cell = tf.contrib.rnn.MultiRNNCell(
                        cells=[tf.contrib.rnn.LSTMCell(
                            num_units=self.config.numHiddenUnits,
                            forget_bias=1.0,
                            state_is_tuple=True)
                               for _ in range(self.config.lstmLayers)],
                        state_is_tuple=True)
                # Initialize states
                # if self.isInitial:
                self.isInitial = False
                self.states = self.cell.zero_state(
                    batch_size=tf.shape(visualFeat)[0],
                    dtype=tf.float32)

            # Mapping visual features via fully connected layer
            visualDim = self.config.numHiddenUnits
            self.w_fc1 = Utils.getTFVariable(
                name='w_fc1',
                shape=[self.config.visualFeatDim, visualDim])
            self.b_fc1 = Utils.getTFVariable(
                name='b_fc1',
                shape=[visualDim])
            self.fc1 = tf.nn.relu(
                features=tf.matmul(visualFeat, self.w_fc1) + self.b_fc1,
                name='fc1')

            # Obtain output
            inputs = self.fc1
            if self.config.isTrain and self.config.isDropout:
                inputs = tf.nn.dropout(
                    x=inputs,
                    keep_prob=self.config.lstmKeepProb)
            # Connect network
            self.outputs, self.states = self.cell(inputs, self.states)

        # Return
        return self.outputs


# The main function of the demo
def main():
    print sys.argv

    # Input feature
    x_feat = tf.truncated_normal(
        shape=[64, 256],
        mean=0,
        stddev=0.1,
        dtype=tf.float32)
    # Create object
    lstmRepresentationExtractor = LSTMRepresentation(config=Config(visualFeatDim=256))
    # Build graph
    for i in range(0, 20):
        print 'i = ', i
        if i > 0:
            with tf.variable_scope("my_graph", reuse=True):
                temp = lstmRepresentationExtractor.buildGraph(x_feat)
        else:
            with tf.variable_scope("my_graph"):
                temp = lstmRepresentationExtractor.buildGraph(x_feat)
    lstmRepresent = temp
    # Initial op
    initVar = tf.initialize_all_variables()

    merged_summaries = tf.merge_all_summaries()

    with tf.Session() as sess:
        # Initialize variables
        sess.run(initVar)
        # Run graph
        lstmRep= sess.run(lstmRepresent)
        # Print
        print lstmRep

        homeDir = '/home/jielyu/Workspace/Python/AttentionModel'
        summaryDir = os.path.join(homeDir, 'tf_graph/name_scope_2')
        writer = tf.train.SummaryWriter(
            summaryDir,
            graph=tf.get_default_graph())
        writer.close()

# The entry of the demo
if __name__ == '__main__':
    main()