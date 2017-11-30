# encoding: utf8

import os
import sys

import pickle
import numpy as np
import matplotlib.pyplot as plt


class RAMEvaluator(object):

    def __init__(self):
        pass

    @staticmethod
    def drawProcess(image,
                    gtBbox=None,
                    gtLabel=None,
                    pointList=None,
                    preBboxList=None,
                    preLabelList=None,
                    isDrawAllPoints=True,
                    isDrawAllBboxes=False,
                    savePath=None,
                    fontSize=20):
        """
        Function: Draw all steps on the same image
        :param image: 
        :param gtBbox: 
        :param gtLabel: 
        :param pointList: 
        :param preYXList: 
        :param preHWList: 
        :param preLabel: 
        :param isDrawAllPoints: 
        :param isDrawAllBboxes: 
        :return: 
        """
        # Convert image to 3 channels
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if len(image.shape) == 2:
            img = np.zeros(shape=[image.shape[0], image.shape[1], 3],
                           dtype=image.dtype)
            img[:, :, 0] = image
            img[:, :, 1] = image
            img[:, :, 2] = image
        else:
            img = image

        fontdict = {'size': fontSize}
        # Plot image
        # Not display axis
        plt.clf()
        plt.imshow(img)
        plt.axis([0, img.shape[1]-1, img.shape[0]-1, 0])
        plt.xticks(())
        plt.yticks(())
        # Plot ground truth bbox
        if gtBbox is not None:
            for i in range(0, gtBbox.shape[0]):
                RAMEvaluator.plotRect(gtBbox[i, 0], gtBbox[i, 1],
                                      gtBbox[i, 2], gtBbox[i, 3])
        # Plot ground truth label
        if gtLabel is not None:
            plt.text(x=1, y=3, s='true:'+str(gtLabel),
                     color='red', fontdict=fontdict)

        # Plot fixation points
        if pointList is not None:
            numStep = len(pointList)-1
            start = numStep - 1
            if isDrawAllPoints:
                start = 0
            for step in range(start, numStep):
                point = pointList[step]
                # point = reversePoint(imageSize, point)
                plt.plot(point[0], point[1], 'yo')
                plt.text(x=point[0], y=point[1], s='P:'+str(step+1),
                         color='yellow', fontdict={'size': 16})

        # Plot predicted bbox
        # if preYXList is not None and preHWList is not None:
        if preBboxList is not None:
            start = len(preBboxList)-1
            if isDrawAllBboxes:
                start = 0
            for step in range(start, len(preBboxList)):
                bbox = preBboxList[step][0, :]
                # Plot
                RAMEvaluator.plotRect(bbox[0], bbox[1], bbox[2], bbox[3],
                                      color='green')
                plt.text(x=bbox[0], y=bbox[1], s='step:'+str(step+1),
                         color='green', fontdict=fontdict)
            # score = '%1.2f'%(preBboxList[-1][0, 4])
            # plt.text(x=30, y=-1, s='score:' + score,
            #          color='green', fontdict=fontdict)
        # Plot predicted label
        if preLabelList is not None:
            plt.text(x=20, y=3, s='predict:' + str(preLabelList[-1]),
                     color='green', fontdict=fontdict)
        # Save to file or display on the screen
        if savePath is not None:
            # Get name and extension
            t = savePath.split('.')
            name = ''
            for k in range(len(t) - 1):
                name = name + t[k]
            ext = t[-1]
            foo_fig = plt.gcf()  # 'get current figure
            foo_fig.savefig(savePath, format=ext, dpi=128,
                            bbox_inches='tight')
        else:
            plt.show()

    @staticmethod
    def drawProcessSeq(image,
                       gtBbox=None,
                       gtLabel=None,
                       pointList=None,
                       preBboxList=None,
                       preLabelList=None,
                       savePath=None,
                       fontSize=20):
        """
        Function: Draw all steps on a sequence of images
        :param image: 
        :param gtBbox: 
        :param gtLabel: 
        :param pointList: 
        :param preYXList: 
        :param preHWList: 
        :param preLabel: 
        :return: 
        """
        # Convert image to 3 channels
        if len(image.shape) == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if len(image.shape) == 2:
            img = np.zeros(shape=[image.shape[0], image.shape[1], 3],
                           dtype=image.dtype)
            img[:, :, 0] = image
            img[:, :, 1] = image
            img[:, :, 2] = image
        else:
            img = image

        fontdict = {'size': fontSize}
        # The number of steps
        numStep = 1
        if pointList is not None:
            numStep = len(pointList)-1
        for step in range(0, numStep):
            # Plot image
            plt.clf()
            line = plt.imshow(img)
            plt.axis([0, img.shape[1]-1, img.shape[0]-1, 0])
            plt.xticks(())
            plt.yticks(())
            if img.shape[1] == 800:
                plt.text(x=10, y=40, s='t = ' + str(step+1),
                         color='black', fontdict=fontdict)
            else:
                plt.text(x=1, y=3, s='t = ' + str(step + 1),
                         color='white', fontdict=fontdict)
            # Plot ground truth bbox
            if gtBbox is not None:
                for i in range(0, gtBbox.shape[0]):
                    RAMEvaluator.plotRect(gtBbox[i, 0], gtBbox[i, 1],
                                          gtBbox[i, 2], gtBbox[i, 3])
            # Plot ground truth label
            if gtLabel is not None:
                plt.text(x=13, y=3, s='true:' + str(gtLabel),
                         color='red', fontdict=fontdict)

            # Plot fixation points
            if pointList is not None:
                point = pointList[step]
                plt.plot(point[0], point[1], 'yo')
                plt.text(x=point[0], y=point[1], s='P:' + str(step + 1),
                         color='yellow', fontdict=fontdict)
                # if img.shape[1] == 800:
                if step != 0:
                    for t_step in range(0, step):
                        lastPt = pointList[t_step]
                        curPt = pointList[t_step+1]
                        line.axes.annotate('',
                                           xytext=(lastPt[0], lastPt[1]),
                                           xy=(curPt[0], curPt[1]),
                                           arrowprops=dict(arrowstyle="->", color='yellow'),
                                           size=15)
                #
                # else:
                #     if step != 0:
                #         for t_step in range(0, step):
                #             lastPt = pointList[t_step]
                #             curPt = pointList[t_step + 1]
                #             line.axes.annotate('',
                #                                xytext=(lastPt[0], lastPt[1]),
                #                                xy=(curPt[0], curPt[1]),
                #                                arrowprops=dict(arrowstyle="->", color='yellow'),
                #                                size=15)

            # Plot predicted bbox
            if preBboxList is not None:
                bbox = preBboxList[step][0, :]
                # Plot
                RAMEvaluator.plotRect(bbox[0], bbox[1], bbox[2], bbox[3],
                                      color='green')
                plt.text(x=bbox[0], y=bbox[1], s='step:' + str(step + 1),
                         color='green', fontdict=fontdict)
                # score = '%1.2f'%(preBboxList[-1][0, 4])
                # plt.text(x=30, y=-1, s='score:'+str(score), color='green')
            # Plot predicted label
            if preLabelList is not None:
                plt.text(x=23, y=3, s='predict:' + str(preLabelList[step]),
                         color='green', fontdict=fontdict)
            # Save to file or display on the screen
            if savePath is not None:
                # print savePath
                # Get name and extension
                t = savePath.split('.')
                name = t[0]
                for k in range(1, len(t)-1):
                    name = name + '.' + t[k]
                ext = t[-1]
                path = name + '_' + str(step) + '.' + ext
                foo_fig = plt.gcf()  # 'get current figure
                foo_fig.savefig(path, format=ext, dpi=128,
                                bbox_inches='tight')
            else:
                plt.show()

    @staticmethod
    def drawMultiProcess(items, indexList=[0], isSeq=False, saveDir=None, ext='png'):

        for i in indexList:
            item = items[i]
            savePath = None
            if saveDir is not None:
                name = '%0*d' % (5, i) + '.' + ext
                savePath = os.path.join(saveDir, name)
            # Draw each process
            if isSeq:
                RAMEvaluator.drawProcessSeq(
                    image=item['image'],
                    gtBbox=item['gtBbox'],
                    gtLabel=item['gtLabel'],
                    pointList=item['pointList'],
                    preBboxList=item['preBboxList'],
                    preLabelList=item['preLabelList'],
                    savePath=savePath)

            else:
                RAMEvaluator.drawProcess(
                    image=item['image'],
                    gtBbox=item['gtBbox'],
                    gtLabel=item['gtLabel'],
                    pointList=item['pointList'],
                    preBboxList=item['preBboxList'],
                    preLabelList=item['preLabelList'],
                    savePath=savePath)

    @staticmethod
    def reverseBbox(imageSize, bbox):
        """
        Function: Convert a bbox yxhw[-1,1][0,2] to xywh[pixel]
        :param imageSize: 
        :param bbox: 
        :return: 
        """
        # Offset
        bbox[0] += 1
        bbox[1] += 1
        # Scale
        t_bbox = [0]*4
        t_bbox[0] = bbox[1] * imageSize[1]/2
        t_bbox[1] = bbox[0] * imageSize[0]/2
        t_bbox[2] = bbox[3] * imageSize[1]/2
        t_bbox[3] = bbox[2] * imageSize[0]/2
        # Return
        return t_bbox

    @staticmethod
    def reversePoint(imageSize, point):
        """
        Function: Convert a point yx[-1, 1] to xy[pixel]
        :param imageSize: 
        :param point: 
        :return: 
        """
        # Scale and Offset
        t_point = [0]*2
        t_point[0] = (point[1]+1)*imageSize[1]/2
        t_point[1] = (point[0]+1)*imageSize[0]/2
        # Return
        return t_point

    @staticmethod
    def plotRect(x, y, w, h, color='red', lw=3):
        """
        Function: Draw a rectangle with specific parameters
        :param x: 
        :param y: 
        :param w: 
        :param h: 
        :param color: 
        :param lw: 
        :return: 
        """
        plt.plot([x, x, x+w, x+w, x],
                 [y, y+h, y+h, y, y],
                 color=color, lw=lw)

    @staticmethod
    def computeOverlap(gtBboxes, preBboxes):
        """
        Function: Compute IoU for objects
        :param gtBboxes: 
        :param preBboxes: 
        :return: 
        """
        # Compute overlap
        # dy = max(min(yp1, yt1) - max(yp0, yt0), 0)
        yp1_yt1 = np.concatenate(
            (np.reshape(preBboxes[:, 1] + preBboxes[:, 3],
                        newshape=[-1, 1]),
             np.reshape(gtBboxes[:, 1] + preBboxes[:, 3],
                        newshape=[-1, 1])), axis=1)
        yp0_yt0 = np.concatenate((np.reshape(preBboxes[:, 1],
                                             newshape=[-1, 1]),
                                  np.reshape(gtBboxes[:, 1],
                                             newshape=[-1, 1])), axis=1)
        delta_y = np.maximum(
            np.min(yp1_yt1, axis=1) - np.max(yp0_yt0, axis=1), 0)

        # dx = max(min(xp1, xt1) - max(xp0, xt0), 0)
        xp1_xt1 = np.concatenate(
            (np.reshape(preBboxes[:, 0] + preBboxes[:, 2],
                        newshape=[-1, 1]),
             np.reshape(gtBboxes[:, 0] + preBboxes[:, 2],
                        newshape=[-1, 1])), axis=1)
        xp0_xt0 = np.concatenate((np.reshape(preBboxes[:, 0],
                                             newshape=[-1, 1]),
                                  np.reshape(gtBboxes[:, 0],
                                             newshape=[-1, 1])), axis=1)
        delta_x = np.maximum(
            np.min(xp1_xt1, axis=1) - np.max(xp0_xt0, axis=1), 0)
        # a = dy*dx
        # r = a/(ga+pa-a)
        overArea = delta_y * delta_x
        preArea = preBboxes[:, 2] * preBboxes[:, 3]
        gtArea = gtBboxes[:, 2] * gtBboxes[:, 3]
        overlap = overArea / (gtArea + preArea - overArea + 1e-8)
        # Return
        return overlap

    @staticmethod
    def evaluate_mAP(gtBboxes,  preBboxes, gtLabels, preLabels,
                     isShowPR=False):
        """
        Function: Compute mAP for multi-class object detection task
        :param gtBboxes: 
        :param preBboxes: 
        :param gtLabels: 
        :param preLabels: 
        :param isShowPR: 
        :return: 
        """
        # # Compute overlap
        overlap = RAMEvaluator.computeOverlap(gtBboxes=gtBboxes,
                                              preBboxes=preBboxes)
        # print overlap.shape
        # print 'overlap = ', np.mean(overlap)
        # Retrieve for each category
        categories = np.unique(gtLabels)
        APList = [0]*categories.shape[0]
        for i in range(0, categories.shape[0]):
            # The number of positive samples
            gt_index = gtLabels == categories[i]
            numPos = np.sum(gt_index)
            # The samples predicted as positive
            pre_index = preLabels == categories[i]
            pre_ol = overlap[pre_index]
            pre_gtLabels = gtLabels[pre_index]
            pre_preLabels = preLabels[pre_index]
            # Compute recall-precise
            thArray = np.arange(0, 1000) / 1000.0
            recallList = []
            preciseList = []
            for j in range(0, thArray.shape[0]):
                th = thArray[j]
                # Select samples
                select_index = pre_ol > th
                numSelect = np.sum(select_index)
                select_gt = pre_gtLabels[select_index]
                select_pre = pre_preLabels[select_index]
                # True positive
                tp_index = np.equal(select_gt, select_pre)
                numTP = float(np.sum(tp_index))
                # Recall and precise
                recallList.append(numTP / (float(numPos) + 1e-8))
                preciseList.append(numTP / (float(numSelect) + 1e-8))

            recall = np.array(recallList, dtype=np.float32)
            precise = np.array(preciseList, dtype=np.float32)
            if isShowPR:
                plt.plot(recall, precise, '-bo')
                plt.axis([0, 1.0, 0, 1.0])
                plt.show()

            # Compute AP
            thArray = np.arange(0, numPos) / float(numPos)
            max_precise = [0]*numPos
            for j in range(0, numPos):
                th = thArray[j]
                # Compute max precise
                rc_index = recall > th
                t_precise = precise[rc_index]
                if t_precise.size == 0:
                    max_precise[j] = 0
                else:
                    max_precise[j] = np.max(t_precise)
            # AP
            APList[i] = np.mean(np.array(max_precise, dtype=np.float32))
        # mAP
        mAP = np.mean(np.array(APList, dtype=np.float32))
        # Return
        return mAP

    @staticmethod
    def evaluate_IoURecall(gtBboxes, preBboxes, savePath=None):
        # Compute overlap
        overlap = RAMEvaluator.computeOverlap(gtBboxes=gtBboxes,
                                              preBboxes=preBboxes)
        thArray = np.arange(0, 1000) / 1000.0
        recallList = [0]*thArray.shape[0]
        for i in range(0, thArray.shape[0]):
            th = thArray[i]
            # Compute recall
            index = overlap > th
            recallList[i] = np.sum(index)/float(overlap.shape[0])
        recall = np.array(recallList, dtype=np.float32)

        plt.plot(thArray, recall, '-bo')
        # Save to file or display on the screen
        if savePath is not None:
            t = savePath.split('.')
            name = t[0]
            for k in range(1, len(t) - 1):
                name = name + '.' + t[k]
            ext = t[-1]
            # name, ext = savePath.split('.')
            foo_fig = plt.gcf()  # 'get current figure
            foo_fig.savefig(savePath, format=ext, dpi=128,
                            bbox_inches='tight')
            txtPath = os.path.join(name + '.txt')
            with open(txtPath, 'w') as fid:
                fid.write('th = \r\n')
                fid.write(str(thArray))
                fid.write('\r\n')
                fid.write('recall = \r\n')
                fid.write(str(recall))
        else:
            plt.show()

    @staticmethod
    def parseBboxesLabelsForSingleObj(items):
        """
        Function: Convert dict to numpy.ndarray
        :param items: 
        :return: 
        """
        # Get the number of images
        num = len(items)
        gtBboxes = np.zeros(shape=[num, 5], dtype=np.float32)
        preBboxes = np.zeros(shape=[num, 5], dtype=np.float32)
        gtLabels = np.zeros(shape=[num], dtype=np.int32)
        preLabels = np.zeros(shape=[num], dtype=np.int32)
        for i in range(0, num):
            item = items[i]
            if 'gtBbox' in item:
                if item['gtBbox'] is not None:
                    gtBboxes[i, 0:4] = item['gtBbox'][0, :]
            if 'preBboxList' in item:
                if item['preBboxList'] is not None:
                    preBboxes[i, :] = item['preBboxList'][-1][0, :]
            if 'gtLabel' in item:
                if item['gtLabel'] is not None:
                    gtLabels[i] = item['gtLabel']
            if 'preLabelList' in item:
                if item['preLabelList'] is not None:
                    preLabels[i] = item['preLabelList'][-1]
        # Check none array
        if np.mean(gtBboxes) < 1e-3:
            gtBboxes = None
        if np.mean(preBboxes) < 1e-3:
            preBboxes = None
        if np.mean(gtLabels) < 1e-3:
            gtLabels = None
        if np.mean(preLabels) < 1e-3:
            preLabels = None
        # Return
        return gtBboxes, preBboxes, gtLabels, preLabels

    @staticmethod
    def parseDict(evalDataDict):
        """
        Function: Convert the original dictionary to item dictionary
        :param evalDataDict: 
        :return: 
        """
        items = []
        # Retrieve all batches
        numBatch = len(evalDataDict['imagesList'])
        for i in range(0, numBatch):
            # Extract data of the i-th image
            imgs = evalDataDict['imagesList'][i]
            gtBboxes = None
            gtLabels = None
            pointsList = None
            preYXsList = None
            preHWsList = None
            preScoresList = None
            preLabelsList = None
            if 'gtBboxesList' in evalDataDict:
                gtBboxes = evalDataDict['gtBboxesList'][i]
            if 'gtLabelsList' in evalDataDict:
                gtLabels = evalDataDict['gtLabelsList'][i]
            if 'pointsListList' in evalDataDict:
                pointsList = evalDataDict['pointsListList'][i]
            if 'preYXListList' in evalDataDict:
                preYXsList = evalDataDict['preYXListList'][i]
            if 'preHWListList' in evalDataDict:
                preHWsList = evalDataDict['preHWListList'][i]
            if 'preScoreListList' in evalDataDict:
                preScoresList = evalDataDict['preScoreListList'][i]
            if 'preLabelsListList' in evalDataDict:
                preLabelsList = evalDataDict['preLabelsListList'][i]
            # Retrieve all images
            for j in range(0, imgs.shape[0]):
                img = imgs[j, :]
                imageSize = img.shape[0:2]
                # Ground truth bbox
                gtBbox = None
                if gtBboxes is not None:
                    gtBbox = gtBboxes[j, :]
                # Ground truth label
                gtLabel = None
                if gtLabels is not None:
                    gtLabel = gtLabels[j]
                # Predict fixation points
                pointList = None
                if pointsList is not None:
                    pointList = []
                    for k in range(0, len(pointsList)):
                        point = pointsList[k][j, :]
                        point = RAMEvaluator.reversePoint(imageSize=imageSize,
                                                          point=point)
                        pointList.append(point)
                # Predict bbox
                preBboxList = None
                if preYXsList is not None and preHWsList is not None \
                        and preScoresList is not None:
                    preBboxList = []
                    for k in range(0, len(preYXsList)):
                        yx = preYXsList[k][j, :]
                        hw = preHWsList[k][j, :]
                        # Reverse
                        bbox = [yx[0], yx[1], hw[0], hw[1]]
                        bbox = RAMEvaluator.reverseBbox(imageSize=imageSize,
                                                        bbox=bbox)
                        bbox.append(preScoresList[k][j][0])
                        bbox = np.array([bbox], dtype=np.float32)
                        preBboxList.append(bbox)
                # Predict label
                preLabelList = None
                if preLabelsList is not None:
                    preLabelList = []
                    for k in range(0, len(preLabelsList)):
                        preLabelList.append(preLabelsList[k][j])
                # Restore to a dictionary
                item = {'image': img,
                        'gtBbox': gtBbox,
                        'gtLabel': gtLabel,
                        'pointList': pointList,
                        'preBboxList': preBboxList,
                        'preLabelList': preLabelList}
                items.append(item)
        # Return all data with dict
        return items


def main():
    print(sys.argv)
    dataDir = 'D:/workspace/Python/EvaluateRAM/data'
    dataName = 'evaluate_data_dict.pkl'
    dataPath = os.path.join(dataDir, dataName)

    with open(dataPath, 'r') as input:
        evalDataDict = pickle.load(input)

    for key, value in evalDataDict.items():
        print(key+': %d'%(len(value)))

    items = RAMEvaluator.parseDict(evalDataDict)
    print(len(items))

    gtBboxes, preBboxes, gtLabels, preLabels = \
        RAMEvaluator.parseBboxesLabelsForSingleObj(items=items)

    RAMEvaluator.evaluate_mAP(gtBboxes=gtBboxes,
                              preBboxes=preBboxes,
                              gtLabels=gtLabels,
                              preLabels=preLabels)

    saveDir = 'D:/workspace/Python/EvaluateRAM/output'
    RAMEvaluator.drawMultiProcess(
        items, range(10), isSeq=True, saveDir=saveDir)


if __name__ == '__main__':
    main()
