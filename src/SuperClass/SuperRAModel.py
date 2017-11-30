# encoding: utf8

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf

from SuperRAMConfig import SuperRAMConfig as Config
from SuperModel import SuperModel


class SuperRAModel(SuperModel):

    def __init__(self, config=Config()):
        super(SuperRAModel, self).__init__(config=config)

        self.pointList = 0  # the focused points of all steps
        self.pointMeanList = 0

        self.reward = 0  # reward
        self.baseline_mse = 0  # mse of baseline
        self.logll = 0  # log likelihood

    def buildEntropyGraph(self, preClaList, gtCla, name='CrossEntropy'):

        if len(preClaList) != self.config.maxTimeStep:
            raise ValueError('Not predicted class')

        ceList = []
        accList = []
        rwdList = []
        with tf.variable_scope(name):
            for i in range(0, self.config.maxTimeStep):
                preCla = preClaList[i]
                # Cross entropy
                ce = tf.reduce_mean(
                        -tf.reduce_sum(
                            gtCla * tf.log(tf.clip_by_value(
                                preCla,
                                1e-10,
                                1.0)),
                            reduction_indices=[1]))
                ceList.append(ce)
                # Reward
                rwd = tf.cast(
                        tf.equal(
                            x=tf.argmax(preCla, 1),
                            y=tf.argmax(gtCla, 1)),
                        tf.float32)
                rwdList.append(rwd)
                # Accuracy
                acc = tf.reduce_mean(rwd)
                accList.append(acc)
        # Return
        return ceList, accList, rwdList

    def buildLogLikelihoodGraph(self, pointList, pointMeanList, name='LogLikeliHood'):
        if len(pointList) > self.config.maxTimeStep:
            pointList = pointList[1:len(pointList)]
            pointMeanList = pointMeanList[1:len(pointMeanList)]

        with tf.variable_scope(name):
            mu = tf.stack(pointMeanList)  # [timestep, batchsize, 2]
            x = tf.stack(pointList)  # [timestep, batchsize, 2]
            gaussian = \
                tf.contrib.distributions.Normal(mu,
                                                self.config.samplingStd)
            logll = tf.log(gaussian.prob(x))
            logll = tf.reduce_sum(logll, 2)  # [timestep, batchsize]
            logll = tf.transpose(logll)  # [batchsize, timestep]

        return logll

    def buildOverlapGraph(self, preYXList, preHWList, gtYXHW, name='Overlap'):
        msrList = []
        overlapList = []
        with tf.variable_scope(name):
            gtYX = gtYXHW[:, 0, 0:2]
            gtHW = gtYXHW[:, 0, 2:4]
            # Get timestep
            timestep = len(preYXList)
            for i in range(0, timestep):
                preYX = preYXList[i]
                preHW = preHWList[i]
                # Compute mean square root of center point
                msr = tf.sqrt(
                    tf.reduce_sum(
                        tf.square((gtYX + gtHW / 2.0) -
                                  (preYX + preHW / 2.0)),
                        reduction_indices=1
                    )
                )
                # Add to list
                msrList.append(msr)

                # Compute overlap
                # dy = max(min(yp1, yt1) - max(yp0, yt0), 0)
                # dx = max(min(xp1, xt1) - max(xp0, xt0), 0)
                delta_y = tf.nn.relu(
                    tf.reduce_min(
                        tf.concat(axis=1,
                                  values=[tf.reshape(preYX[:, 0]+preHW[:, 0],
                                                     shape=[-1, 1]),
                                          tf.reshape(gtYX[:, 0] + gtHW[:, 0],
                                                     shape=[-1, 1])]),
                        reduction_indices=1) -
                    tf.reduce_max(
                        tf.concat(axis=1,
                                  values=[tf.reshape(preYX[:, 0],
                                                     shape=[-1, 1]),
                                          tf.reshape(gtYX[:, 0],
                                                     shape=[-1, 1])]),
                        reduction_indices=1)
                )
                delta_x = tf.nn.relu(
                    tf.reduce_min(
                        tf.concat(axis=1,
                                  values=[tf.reshape(preYX[:, 1]+preHW[:, 1],
                                                     shape=[-1, 1]),
                                          tf.reshape(gtYX[:, 1] + gtHW[:, 1],
                                                     shape=[-1, 1])]),
                        reduction_indices=1) -
                    tf.reduce_max(
                        tf.concat(axis=1,
                                  values=[tf.reshape(preYX[:, 1],
                                                     shape=[-1, 1]),
                                          tf.reshape(gtYX[:, 1],
                                                     shape=[-1, 1])]),
                        reduction_indices=1)
                )
                # a = dy*dx
                # r = a/(ga+pa-a)
                overArea = delta_y * delta_x
                preArea = preHW[:, 0] * preHW[:, 1]
                gtArea = gtHW[:, 0] * gtHW[:, 1]
                overlap = overArea / (gtArea + preArea - overArea)
                # Add to list
                overlapList.append(overlap)

        return msrList, overlapList

    def buildDistanceSqareGraph(self,
                                preYXList,
                                preHWList,
                                gtYXHW,
                                name='DistanceSquare'):
        dsList = []
        with tf.variable_scope(name):
            timestep = len(preYXList)
            gtYX = gtYXHW[:, 0, 0:2]
            gtHW = gtYXHW[:, 0, 2:4]
            for i in range(0, timestep):
                preYX = preYXList[i]
                preHW = preHWList[i]
                preYXHW = tf.concat(axis=1,
                                    values=[preYX, preHW])
                # ds = tf.reduce_sum(
                #     tf.square(gtYXHW - preYXHW),
                #     reduction_indices=1
                # )
                # ds = tf.reduce_sum(
                #     tf.square((gtYX + gtHW/2.0) - (preYX + preHW/2.0)),
                #     reduction_indices=1
                # )
                # Left-up point
                ds = tf.reduce_sum(
                    tf.square(gtYX - preYX),
                    axis=1
                )
                dsList.append(ds)

        return dsList

    def buildAttentionDistanceGraph(self,
                                    pointsList,
                                    gtYXHW,
                                    name='AttentionDistanceSquare'):
        dsList = []
        with tf.variable_scope(name):
            t = len(pointsList)
            gtYX = gtYXHW[:, 0, 0:2]
            gtHW = gtYXHW[:, 0, 2:4]
            for i in range(0, t):
                preYX = pointsList[i]
                ds = tf.reduce_sum(
                    tf.square((gtYX + gtHW/2.0) - preYX),
                    axis=1)
                dsList.append(ds)
        return dsList

    @staticmethod
    def saveRecurrentTrack(saveDir,
                           images,
                           pointsList=None,
                           YXList=None,
                           HWList=None,
                           scoreList=None,
                           preClaName=None,
                           gtBbox=None,
                           gtClaName=None,
                           isDisp=False,
                           isSaveData=False):

        if not isinstance(images, np.ndarray):
            raise TypeError('Not class ndarray')

        if np.max(images) > 0.5 or np.min(images) < -0.5:
            raise ValueError('Not normalized inputs')

        numImgs = images.shape[0]
        h = images.shape[1]
        w = images.shape[2]
        hwArray = np.array([h / 2.0, w / 2.0], dtype=np.float32)

        # Create direction to save those figures
        if not os.path.isdir(saveDir):
            os.makedirs(saveDir)
        # Save data
        if isSaveData:
            t_dict = {'images': images,
                      'pointsList': pointsList,
                      'YXList': YXList,
                      'HWList': HWList,
                      'scoreList': scoreList,
                      'gtBbox': gtBbox}
            resDataPath = os.path.join(saveDir, 'result.data.pkl')
            with open(resDataPath, 'wb') as output:
                pickle.dump(t_dict, output)
        # Draw a series of focused points
        gtBbox = gtBbox.copy()
        for i in range(0, numImgs):

            # Draw image
            t_img = images[i, :]
            if t_img.dtype != np.uint8:
                t_img = (t_img + 0.5) * 255
                t_img = t_img.astype(np.uint8)

            if t_img.shape[2] == 1:
                t_img = t_img[:, :, 0]

            plt.clf()
            plt.imshow(t_img)
            plt.axis([0, w, h, 0])

            # Draw ground truth
            if gtBbox is not None:
                if not isinstance(gtBbox, np.ndarray):
                    raise TypeError('Not class ndarray')
                else:
                    if len(gtBbox.shape) == 3:
                        gtBbox = np.reshape(gtBbox, newshape=[-1, 4])
                    t_bbox = gtBbox[i, :]
                    t_bbox[0:2] = (t_bbox[0:2] + 1) * hwArray
                    t_bbox[2:4] = t_bbox[2:4] * hwArray
                    plt.plot([t_bbox[1], t_bbox[1], t_bbox[1] + t_bbox[3],
                              t_bbox[1] + t_bbox[3], t_bbox[1]],
                             [t_bbox[0], t_bbox[0] + t_bbox[2],
                              t_bbox[0] + t_bbox[2], t_bbox[0], t_bbox[0]],
                             color='red', lw=3)

            # Plot ground-truth class
            if gtClaName is not None:
                t_name = 'gt:' + str(gtClaName[i])
                plt.text(x=0,
                         y=h/4,
                         s=t_name,
                         color='red')
            # Plot predicted class
            if preClaName is not None:
                t_name = 'pre:' + str(preClaName[i])
                plt.text(x=0,
                         y=h*3/4,
                         s=t_name,
                         color='green')

            # Draw focused points
            if pointsList is not None:
                if not isinstance(pointsList, list):
                    raise TypeError('Not a series of points')
                numStep = len(pointsList)
                pts = np.zeros(shape=[numStep, 2], dtype=np.float32)
                for j in range(0, numStep):
                    t_point = pointsList[j][i, :]
                    t_point = (t_point + 1) * hwArray
                    pts[j, :] = np.reshape(t_point, newshape=[1, 2])
                    # plt.plot(t_point[1], t_point[0], 'bo')
                    plt.text(x=t_point[1],
                             y=t_point[0],
                             s=str(j),
                             color='yellow')
                plt.plot(pts[:, 1], pts[:, 0], 'yo')

            # Draw predicted bbox
            if (HWList is not None) and (YXList is not None):
                if len(HWList) != len(YXList):
                    raise ValueError('Not proper bbox')
                for j in range(0, len(HWList)):
                    # # Draw the final one
                    # if j != len(HWList)-1:
                    #     continue
                    t_hw = HWList[j][i, :] * hwArray
                    t_yx = (YXList[j][i, :] + 1) * hwArray
                    s = str(j + 1)
                    if scoreList is not None:
                        if len(scoreList) != len(YXList):
                            raise ValueError('Not proper score')
                        else:
                            t_score = scoreList[j][i]
                            s = s + ': s=%1.2f' % (t_score)
                    plt.text(x=t_yx[1], y=t_yx[0], s=s, color='green')

                    plt.plot([t_yx[1], t_yx[1], t_yx[1] + t_hw[1],
                              t_yx[1] + t_hw[1], t_yx[1]],
                             [t_yx[0], t_yx[0] + t_hw[0],
                              t_yx[0] + t_hw[0], t_yx[0], t_yx[0]],
                             color='green', lw=3)

            # Set legend
            plt.legend(['ground truth', 'focused points', 'predicted'],
                       fontsize='x-small')
            # Save figure to file
            fileName = '%05d' % (i)
            fileName += '.png'
            filePath = os.path.join(saveDir, fileName)
            foo_fig = plt.gcf()  # 'get current figure
            foo_fig.savefig(filePath,
                            format='png',
                            dpi=128,
                            bbox_inches='tight')

            # Display
            if isDisp:
                plt.show()