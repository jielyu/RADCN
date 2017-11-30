# encoding:utf-8

"""
Author: Jie Lyu
E-mail: jiejielyu@outlook.com
Date: 2017.10.23
"""

# System libraries
import os
import skimage.io
import sys
import time

import numpy as np
from matplotlib import pyplot as plt

from src.SuperClass.SuperDataset import SuperDataset
from src.SuperClass.SuperDataset import SuperDatasetManager


# Define and implement a class
# to manage dataset
class DatasetManager(SuperDatasetManager):

    def __init__(self,
                 enableMemSave=False,
                 isRandomBatch=True,
                 batchSize=64,
                 bboxMaxNum=1,
                 maxSampleNum=0):
        """
        Function: Initialize an object
        :param batchSize: the size of each batch
        :param bboxMaxNum: the maximum number of bbox
        Author: Jie Lyu
        Date: 2017.01.23
        """
        super(DatasetManager, self).__init__()
        # Enable memory save model
        self.enableMemSave = enableMemSave
        self.isRandomBatch = isRandomBatch
        self.imagePathList = []
        self.bboxPathList = []
        # Data
        self.images = 0  # [n, h, w, ch]
        self.bbox = 0    # [n, m, 4],(x,y,w,h)
        self.bboxMaxNum = bboxMaxNum

        # Counter
        self.batchSize = batchSize
        self.maxSampleNum = maxSampleNum
        self.batchCount = 0
        self.epochCount = 0
        self.randomIndex = 0

    @staticmethod
    def readImage(path):
        """
        Function: Read image from the specific path
        :param path: The path of an image
        :return:
        """
        return skimage.io.imread(fname=path)

    @staticmethod
    def readBbox(path):
        """
        Function: Read bbox from the specific path
        :param path: The path of bbox text file
        :return:
        """

        with open(path, 'r') as fid:

            # Check the header of the file
            line = fid.readline()
            if not line == '# Object Bbox\r\n':
                raise ValueError('Not bbox file')

            # Read the number of bbox
            line = fid.readline()
            num = line.split()
            num = int(num[1])
            # Check the number of bbox
            if num > 0:
                # Check the parameters of the file
                line = fid.readline()
                if not line == 'x-y-w-h:\r\n':
                    raise ValueError('Not x-y-w-h bbox')

                # Read the parameters of each bbox
                bbox = np.zeros(shape=[num, 4], dtype=np.float32)
                for i in range(0, num):
                    line = fid.readline()
                    line = line.split()
                    if not len(line) == 4:
                        raise ValueError('Not 4 parameters bbox')
                    # Convert string to float
                    bbox[i, 0] = float(line[0])
                    bbox[i, 1] = float(line[1])
                    bbox[i, 2] = float(line[2])
                    bbox[i, 3] = float(line[3])
            else:
                bbox = np.array([], dtype=np.float32)

        # Return bbox
        return bbox

    def readDataset(self, txtPath):
        """
        Function: Read dataset from the specific path
        :param txtPath:
        :return:  None
        """
        # Check the existence
        if not os.path.isfile(txtPath):
            raise ValueError('Not a text file')

        # Get direction
        fdir, fname = os.path.split(txtPath)

        # Read paths of images and corresponding bbox
        imgName = []
        bboxName = []
        with open(txtPath, 'r') as fid:
            # Read the first line
            line = fid.readline()
            while line:
                line = line.split()
                if len(line) == 1:
                    imgName.append(line[0])
                    bboxName.append('')
                elif len(line) == 2:
                    imgName.append(line[0])
                    bboxName.append(line[1])
                else:
                    raise ValueError('Not image paths file')
                # Read next line
                line = fid.readline()

        # Get the number of samples and size of images
        num = len(imgName)
        if self.maxSampleNum != 0 and num > self.maxSampleNum:
            num = self.maxSampleNum
        tpath = os.path.join(fdir, imgName[0])
        img = self.readImage(tpath)
        imgShape = img.shape
        if len(imgShape) < 3:
            raise ValueError('Not an w-h-ch image')
        # print num
        # print imgShape

        if self.enableMemSave:
            for i in range(0, num):
                # Concatenate path
                tImgPath = os.path.join(fdir, imgName[i])
                tBboxPath = os.path.join(fdir, bboxName[i])
                self.imagePathList.append(tImgPath)
                self.bboxPathList.append(tBboxPath)
        else:
            # Create memory
            self.images = np.zeros(
                shape=[num, imgShape[0], imgShape[1], imgShape[2]],
                dtype=np.uint8)
            self.bbox = np.zeros(
                shape=[num, self.bboxMaxNum, 4],
                dtype=np.float32)

            # Read data
            for i in range(0, num):
                # Concatenate path
                tImgPath = os.path.join(fdir, imgName[i])
                tBboxPath = os.path.join(fdir, bboxName[i])

                # Check path and load image
                if not os.path.isfile(tImgPath):
                    raise ValueError('Not an Image')
                else:
                    img = self.readImage(tImgPath)
                    self.images[i, :] = img

                # Check path and load bbox
                if os.path.isfile(tBboxPath):
                    bbox = self.readBbox(tBboxPath)
                    numBbox = bbox.shape[0]
                    # Not more than the maximum and Not less than 1
                    if numBbox > self.bboxMaxNum:
                        raise ValueError('More than one bbox')
                    elif numBbox != 0:
                        self.bbox[i, 0:numBbox, :] = bbox

                # Limit the number of samples
                if self.maxSampleNum != 0:
                    if i >= self.maxSampleNum:
                        break

    def getNextBatch(self, batchSize=0):
        """
        Function: output a batch of data with size 64 samples
        :param batchSize: the size of each batch
        :return: images, bboxes
        """

        # Check parameters
        if batchSize != 0 and self.batchSize != batchSize:
            self.batchSize = batchSize
        if self.batchSize < 1:
            raise ValueError('Batch size is less than one')

        # Set batch size
        num = self.getSampleNum()
        if num < self.batchSize:
            raise ValueError('Batch size is more than samples')

        # Create random index
        if self.batchCount == 0:
            if self.isRandomBatch:
                self.randomIndex = \
                    np.random.randint(0, num, num, dtype=np.int32)
            else:
                self.randomIndex = \
                    np.arange(0, num, dtype=np.int32)

        # Compute start and end index
        startIndex = self.batchCount*self.batchSize
        endIndex = (self.batchCount+1)*self.batchSize
        if endIndex > num:
            endIndex = num
            startIndex = num - self.batchSize
            self.batchCount = 0
            self.epochCount += 1
        else:
            self.batchCount += 1

        if self.enableMemSave:
            images, bbox = self.readBatchDataset(self.randomIndex[startIndex:endIndex])
        else:

            # Fetch images and bboxes
            if isinstance(self.images,int) or isinstance(self.bbox,int):
                raise ValueError('Empty Dataset')
            images = self.images[self.randomIndex[startIndex:endIndex], :]
            bbox = self.bbox[self.randomIndex[startIndex:endIndex], :]

        # Return images and bboxes
        return images, bbox

    def readBatchDataset(self, index):
        # Get the number of images
        numImg = index.shape[0]
        # Get the size of images
        tpath = self.imagePathList[0]
        timg = self.readImage(tpath)
        # Create memory
        images = np.zeros(
            shape=[numImg, timg.shape[0], timg.shape[1], timg.shape[2]],
            dtype=np.uint8)
        bboxes = np.zeros(
            shape=[numImg, self.bboxMaxNum, 4],
            dtype=np.float32)
        for i in range(0, numImg):
            # Get path
            tImgPath = self.imagePathList[index[i]]
            tBboxPath = self.bboxPathList[index[i]]
            # Read and store Image
            img = self.readImage(tImgPath)
            images[i, :] = img
            # Read and store Bbox
            bbox = self.readBbox(tBboxPath)
            if bbox.shape[0] > self.bboxMaxNum:
                raise ValueError('More than one bbox')
            bboxes[i, 0:bbox.shape[0], :] = bbox

        return images, bboxes

    def getSampleNum(self):
        # Check memory and return number of samples
        if isinstance(self.images, np.ndarray):
            return self.images.shape[0]
        elif len(self.imagePathList) > 0:
            return len(self.imagePathList)
        else:
            return 0

    def showImage(self, index=0):
        plt.imshow(self.images[index, :])
        plt.show()

    def showBbox(self, index=0):
        if self.enableMemSave and isinstance(self.images, int):
            return
        plt.imshow(self.images[index, :])
        bbox = self.bbox[index, :]
        # Plot five points
        plt.plot([bbox[0, 0], bbox[0, 0] + bbox[0, 2],
                  bbox[0, 0] + bbox[0, 2], bbox[0, 0], bbox[0, 0]],
                 [bbox[0, 1], bbox[0, 1], bbox[0, 1] + bbox[0, 3],
                  bbox[0, 1] + bbox[0, 3], bbox[0, 1]], color='red', lw=3)
        plt.show()


# Define and implement a class
# to configure parameters of FCARDataset
class Config:

    def __init__(self,
                 enableMemSave=False,
                 isAllTime=False,
                 isDayTime=True,
                 bboxMaxNum=1,
                 batchSize=64,
                 maxSampleNum=0,
                 testingSampleRatio=0.3,
                 datasetDir='.'):
        """
        Function: Initialize the parameters of configuration
        :param isAllTime: Enable all-time flag
        :param isDayTime: Enable day-time flag
        :param bboxMaxNum:  The maximum number of bbox
        :param batchSize:   The size of each batch
        :param datasetDir:  The direction of dataset
        """
        self.enableMemSave = enableMemSave
        # Set scenario flag
        self.isAllTime = isAllTime
        self.isDayTime = isDayTime
        # Set maximum number of bbox
        self.bboxMaxNum = bboxMaxNum
        # Set batch size
        self.batchSize = batchSize
        self.maxSampleNum = maxSampleNum
        self.testingSampleRatio = testingSampleRatio
        # Set direction of dataset
        self.datasetDir = datasetDir
        self.teTrName = ['testset', 'trainset']
        self.dayNight = ['daytime', 'night']


# Define and implement a class to
# package operations related to FCAR-dataset
class FCARDataset(SuperDataset):

    def __init__(self, config=Config()):
        super(FCARDataset, self).__init__()
        # Configuration
        self.config = config

        # Dataset
        maxNum = np.int(config.maxSampleNum * config.testingSampleRatio)
        self.testset = DatasetManager(
            enableMemSave=self.config.enableMemSave,
            isRandomBatch=False,
            batchSize=self.config.batchSize,
            bboxMaxNum=self.config.bboxMaxNum,
            maxSampleNum=maxNum)
        maxNum = np.int(config.maxSampleNum * (1 - config.testingSampleRatio))
        self.trainset = DatasetManager(
            enableMemSave=self.config.enableMemSave,
            batchSize=self.config.batchSize,
            bboxMaxNum=self.config.bboxMaxNum,
            maxSampleNum=maxNum)

    def readDataset(self, isTrain=True, isTest=True):
        # Compute direction of testset and trainset
        testsetDir = os.path.join(
            self.config.datasetDir,
            self.config.teTrName[0])
        trainsetDir = os.path.join(
            self.config.datasetDir,
            self.config.teTrName[1])

        # All-time scenario
        if self.config.isAllTime:
            testsetTxtPath = os.path.join(testsetDir,
                                           self.config.teTrName[0]+'.txt')
            trainsetTxtPath = os.path.join(trainsetDir,
                                           self.config.teTrName[1]+'.txt')
        else:
            # Day-time scenario
            dayNight = self.config.dayNight
            if self.config.isDayTime:
                testsetTxtPath = \
                    os.path.join(testsetDir,
                                 dayNight[0],
                                 dayNight[0]+'.txt')
                trainsetTxtPath = \
                    os.path.join(trainsetDir,
                                 dayNight[0],
                                 dayNight[0]+'.txt')
            # Night scenario
            else:
                testsetTxtPath = \
                    os.path.join(testsetDir,
                                 dayNight[1],
                                 dayNight[1]+'.txt')
                trainsetTxtPath = \
                    os.path.join(trainsetDir,
                                 dayNight[1],
                                 dayNight[1]+'.txt')
        # print 'tesetsetTxtPath = ', tesetsetTxtPath
        # print 'trainsetTxtPath = ', trainsetTxtPath

        # Check Path
        if not os.path.isfile(testsetTxtPath):
            raise ValueError('Not testset:\r\n\t%s'%(testsetTxtPath))
        if not os.path.isfile(trainsetTxtPath):
            raise ValueError('Not trainset:\r\n\t%s'%(trainsetTxtPath))

        # Read images and bboxes
        start = time.clock()
        if isTest:
            print 'Reading FCAR-dataset testset ...'
            self.testset.readDataset(testsetTxtPath)
        if isTrain:
            print 'Reading FCAR-dataset trainset ...'
            self.trainset.readDataset(trainsetTxtPath)
        end = time.clock()
        print 'Read FCAR-dataset completely, cost time %f seconds' \
              %(end-start)


# The main function of the demo
def main():

    print sys.argv
    datasetHomeDir = '/home/jielyu/Database/FCAR-dataset'
    # Check DatasetManager
    data = DatasetManager(enableMemSave=True)

    # # Read bbox
    # bboxPath = 'testset/daytime/posGt/I_00000001.bbox'
    # bbox = data.readBbox(os.path.join(datasetHomeDir, bboxPath))
    # print 'bbox = ', bbox
    #
    # # Read image
    # imgPath = 'testset/daytime/pos/I_00000001.jpg'
    # img = data.readImage(os.path.join(datasetHomeDir, imgPath))
    # plt.imshow(img)
    # axes = plt.subplot(111)
    # plt.plot([bbox[0, 0], bbox[0, 0]+bbox[0, 2],
    #           bbox[0, 0]+bbox[0, 2], bbox[0, 0], bbox[0, 0]],
    #          [bbox[0, 1], bbox[0, 1], bbox[0, 1]+bbox[0, 3],
    #           bbox[0, 1]+bbox[0, 3], bbox[0, 1]], color='red', lw=3)
    # plt.axis([800, 0, 800, 0])
    # axes.set_xticks([])
    # axes.set_yticks([])
    # plt.show()

    # Read dataset
    print 'numSamples = ', data.getSampleNum()
    testsetPath = 'testset/daytime/daytime.txt'
    data.readDataset(os.path.join(datasetHomeDir, testsetPath))
    print 'numSamples = ', data.getSampleNum()
    data.showBbox(6)
    # Get next batch
    imgs, bboxes = data.getNextBatch()
    print 'imgs.shape = ', imgs.shape
    print 'bboxes.shape', bboxes.shape
    plt.imshow(imgs[0, :])
    plt.show()

    # # Check FCAR-dataset
    config = Config(
        enableMemSave=True,
        isAllTime=False,
        datasetDir=datasetHomeDir)
    fcar = FCARDataset(config=config)
    print isinstance(fcar, SuperDataset)
    fcar.readDataset()
    imgs, bboxes = fcar.trainset.getNextBatch()
    print imgs.shape
    print bboxes.shape
    #
    print 'numSampleOnTestset = ', fcar.testset.getSampleNum()
    print 'numSampleOnTrainset = ', fcar.trainset.getSampleNum()

# The entry of the demo
if __name__ == '__main__':

    main()
