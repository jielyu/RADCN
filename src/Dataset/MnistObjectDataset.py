# encoding:utf-8

"""
Author: Jie Lyu
E-mail: jiejielyu@outlook.com
Date: 2017.10.23
"""

# System libraries
import os
import sys
import time

# 3rd-part libraries
import pickle
import gzip
import numpy
from matplotlib import pyplot as plt
from scipy import misc

from FCARDataset import DatasetManager
from FCARDataset import FCARDataset


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
        f: A file object that can be passed into a gzip reader.

    Returns:
        data: A 4D unit8 numpy array [index, y, x, depth].

    Raises:
        ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
        f: A file object that can be passed into a gzip reader.
        one_hot: Does one hot encoding for the result.
        num_classes: Number of classes for the one hot encoding.

    Returns:
        labels: a 1D unit8 numpy array.

    Raises:
        ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels, num_classes)
        return labels


class MnistManager(DatasetManager):

    def __init__(self,
                 isRandomBatch=True,
                 batchSize=64,
                 bboxMaxNum=1,
                 maxSampleNum=0):
        super(MnistManager, self).__init__(
            isRandomBatch=isRandomBatch,
            batchSize=batchSize,
            bboxMaxNum=bboxMaxNum,
            maxSampleNum=maxSampleNum)
        # self.imageSize = imageSize
        self.labels = 0

    def readDataset(self, txtPath):
        """
        Function: Read dataset from pkl file
        :param txtPath: the path of pkl file
        :return: None
        """
        with open(txtPath, 'rb') as input:
            dataset = pickle.load(input)
            self.images = dataset['images']
            self.bbox = dataset['bbox']
            self.labels = dataset['labels']

            num = self.images.shape[0]
            if num > self.maxSampleNum:
                num = self.maxSampleNum
                self.images = self.images[0: num, :]
                self.bbox = self.bbox[0: num, :]
                self.labels = self.labels[0: num, :]

    def getNextBatchWithLabels(self, batchSize=0):
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
                    numpy.random.randint(0, num, num, dtype=numpy.int32)
            else:
                self.randomIndex = \
                    numpy.arange(0, num, dtype=numpy.int32)

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

        # Fetch images and bboxes
        if isinstance(self.images,int) or isinstance(self.bbox,int):
            raise ValueError('Empty Dataset')
        images = self.images[self.randomIndex[startIndex:endIndex], :]
        bbox = self.bbox[self.randomIndex[startIndex:endIndex], :]
        labels = self.labels[self.randomIndex[startIndex:endIndex], :]
        # Return images and bboxes
        return images, bbox, labels

    def createDataset(self, images, labels, dstPath, imageSize,
                      scaleRange=None,
                      isNoise=False,
                      noiseNum=6,
                      noiseSize=[6, 6]):
        """
        Function: Create dataset and store in pkl file
        :param images:  original images, 4D, [batchsize, h, w, ch]
        :param labels:  labels, 2D, [batchsize, numCategories]
        :param dstPath: storing path
        :param scaleRange:  the range of scale
        :return: None
        """
        self.imageSize = imageSize
        # Check parameter
        if not (isinstance(images, numpy.ndarray)
                and isinstance(labels, numpy.ndarray)):
            raise TypeError('Not np.ndarray')
        if len(images.shape) != 4 or len(labels.shape) != 2:
            raise ValueError('Not images or one-hot labels')
        if (not isinstance(self.imageSize, list)) \
                or (len(self.imageSize) != 2):
            raise TypeError('Wrong image size')
        # Create images
        num = images.shape[0]
        shape = [num, self.imageSize[0], self.imageSize[1], images.shape[3]]
        self.images = numpy.zeros(dtype=numpy.uint8,
                                  shape=shape)
        self.bbox = numpy.zeros(dtype=numpy.float32,
                                shape=[num, self.bboxMaxNum, 4])

        if self.imageSize[0]<=images.shape[1] \
                or self.imageSize[1]<=images.shape[2]:
            self.images = images
            self.bbox[:, :, 2] = images.shape[1]
            self.bbox[:, :, 3] = images.shape[2]
        else:
            for i in range(0, num):
                if images.shape[3] == 1:
                    img = images[i, :, :, 0]
                else:
                    img = images[i, :]
                if scaleRange is not None:
                    if not (isinstance(scaleRange, list) and len(scaleRange)==2):
                        raise TypeError('Not scale range')
                    minScale, maxScale = scaleRange[0], scaleRange[1]
                    scale = numpy.random.uniform(minScale, maxScale)
                    img = misc.imresize(img, scale)
                    # plt.clf()
                    # plt.imshow(img)
                    # plt.show()

                # Compute range
                x_range = self.imageSize[1] - img.shape[1]
                y_range = self.imageSize[0] - img.shape[0]
                # Select position
                x = numpy.random.randint(0, x_range, 1)
                x = x[0]
                y = numpy.random.randint(0, y_range, 1)
                y = y[0]
                w, h = img.shape[1], img.shape[0]
                # Assign
                x_max = x+w
                y_max = y+h
                if images.shape[3] == 3:
                    self.images[i, y:y_max, x: x_max, :] = img
                else:
                    self.images[i, y:y_max, x: x_max, 0] = img
                self.bbox[i, 0, :] = numpy.array([x, y, w, h],
                                                 dtype=numpy.float32)

                if isNoise:
                    self.images[i, :] = self.corruptImage(self.images[i, :],
                                                          noiseSrcImg=images,
                                                          noiseNum=noiseNum,
                                                          noiseSize=noiseSize)

                # plt.clf()
                # img = self.images[i, :, :, 0]
                # plt.imshow(img)
                # plt.show()

        # Store into pickle
        dict = {'images': self.images, 'bbox': self.bbox, 'labels': labels}
        if os.path.isfile(dstPath):
            print dstPath
            raise Warning('Existed file will be rewritten')
        else:
            with open(dstPath, 'wb') as output:
                pickle.dump(dict, output)

    def corruptImage(self, img, noiseSrcImg, noiseNum=6, noiseSize=[6, 6]):

        for i in range(0, noiseNum):
            # Crop patch with random location
            t_imgIndex = numpy.random.randint(0, noiseSrcImg.shape[0])
            nsi = noiseSrcImg[t_imgIndex, :]
            maxRngH = nsi.shape[0] - noiseSize[0]
            maxRngW = nsi.shape[1] - noiseSize[1]
            s_randH = numpy.random.randint(0, maxRngH)
            s_maxH = s_randH+noiseSize[0]
            s_randW = numpy.random.randint(0, maxRngW)
            s_maxW = s_randW + noiseSize[1]
            s_img = nsi[s_randH:s_maxH, s_randW:s_maxW, :]
            # Set patch with random location
            maxRngH = img.shape[0] - noiseSize[0]
            maxRngW = img.shape[1] - noiseSize[1]
            t_randH = numpy.random.randint(0, maxRngH)
            t_maxH = t_randH+noiseSize[0]
            t_randW = numpy.random.randint(0, maxRngW)
            t_maxW = t_randW+noiseSize[1]
            img[t_randH:t_maxH, t_randW:t_maxW, :] = s_img
        # Copy
        t_img = numpy.copy(img)
        return t_img

    def showBbox(self, index=0):
        plt.imshow(self.images[index, :, :, 0])
        bbox = self.bbox[index, :]
        print bbox
        # Plot five points
        plt.plot([bbox[0, 0], bbox[0, 0] + bbox[0, 2],
                  bbox[0, 0] + bbox[0, 2], bbox[0, 0], bbox[0, 0]],
                 [bbox[0, 1], bbox[0, 1], bbox[0, 1] + bbox[0, 3],
                  bbox[0, 1] + bbox[0, 3], bbox[0, 1]], color='red', lw=3)
        plt.show()


class Config:

    def __init__(self,
                 bboxMaxNum=1,
                 batchSize=64,
                 maxSampleNum=100000,
                 testingSampleRatio=0.3,
                 # imageSize = [56, 56],
                 mnistDataDir='.',
                 datasetDir='.'):
        """
        Function: Initialize the parameters of configuration
        :param isAllTime: Enable all-time flag
        :param isDayTime: Enable day-time flag
        :param bboxMaxNum:  The maximum number of bbox
        :param batchSize:   The size of each batch
        :param datasetDir:  The direction of dataset
        """
        self.enableMemSave = False
        # Set maximum number of bbox
        self.bboxMaxNum = bboxMaxNum
        # Set batch size
        self.batchSize = batchSize
        self.maxSampleNum = maxSampleNum
        self.testingSampleRatio = testingSampleRatio
        # self.imageSize = imageSize
        # Set direction of dataset
        self.mnistDataDir = mnistDataDir
        self.datasetDir = datasetDir
        self.teTrName = ['testset.pkl', 'trainset.pkl']
        self.mnistFileName = ['t10k-images-idx3-ubyte.gz',
                              't10k-labels-idx1-ubyte.gz',
                              'train-images-idx3-ubyte.gz',
                              'train-labels-idx1-ubyte.gz']


class MnistObjectDataset(FCARDataset):
    
    def __init__(self, config=Config()):
        super(MnistObjectDataset, self).__init__(config)

        self.config = config
        maxNum = numpy.int(config.maxSampleNum * config.testingSampleRatio)
        self.testset = MnistManager(
            isRandomBatch=False,
            batchSize=self.config.batchSize,
            bboxMaxNum=self.config.bboxMaxNum,
            maxSampleNum=maxNum)
        maxNum = numpy.int(config.maxSampleNum * (1 - config.testingSampleRatio))
        self.trainset = MnistManager(
            batchSize=self.config.batchSize,
            bboxMaxNum=self.config.bboxMaxNum,
            maxSampleNum=maxNum)

    def readDataset(self,
                    isTrain=True,
                    isTest=True):
        """
        Function: Read dataset from pkl files
        :param isTrain: enable trainset
        :param isTest:  enable testset
        :return: None
        """
        if not os.path.isdir(self.config.datasetDir):
            os.makedirs(self.config.datasetDir)

        start = time.clock()
        if isTest:
            print 'Reading MnistObject-dataset testset ...'
            testPath = os.path.join(self.config.datasetDir,
                                    self.config.teTrName[0])
            if not os.path.isfile(testPath):
                raise ValueError('Not existing file')
            self.testset.readDataset(testPath)

        if isTrain:
            print 'Reading MnistObject-dataset trainset ...'
            trainPath = os.path.join(self.config.datasetDir,
                                     self.config.teTrName[1])
            if not os.path.isfile(trainPath):
                raise ValueError('Not existing file')
            self.trainset.readDataset(trainPath)

        end = time.clock()
        print 'Read MnistObject-dataset completely, cost time %f seconds' \
              % (end - start)

    def createDataset(self, isTrain=True, isTest=True, imageSize=[56, 56], scaleRange=None, isNoise=False, noiseNum=6, noiseSize=[6, 6]):
        """
        FUnction: Create dataset with specific parameters
        :param isTrain: enable trainset
        :param isTest:  enable testset
        :param scaleRange:  set range of scale
        :return: None
        """

        if not os.path.isdir(self.config.datasetDir):
            os.makedirs(self.config.datasetDir)
        if not os.path.isdir(self.config.mnistDataDir):
            raise ValueError('Not existing path')

        # Load original mnist data
        local_file = os.path.join(self.config.mnistDataDir,
                                  self.config.mnistFileName[0])
        if not os.path.isfile(local_file):
            raise ValueError('Not existing path')
        with open(local_file, 'rb') as f:
            test_images = extract_images(f)
        local_file = os.path.join(self.config.mnistDataDir,
                                  self.config.mnistFileName[1])
        if not os.path.isfile(local_file):
            raise ValueError('Not existing path')
        with open(local_file, 'rb') as f:
            test_labels = extract_labels(f, one_hot=True)
        local_file = os.path.join(self.config.mnistDataDir,
                                  self.config.mnistFileName[2])
        if not os.path.isfile(local_file):
            raise ValueError('Not existing path')
        with open(local_file, 'rb') as f:
            train_images = extract_images(f)
        local_file = os.path.join(self.config.mnistDataDir,
                                  self.config.mnistFileName[3])
        if not os.path.isfile(local_file):
            raise ValueError('Not existing path')
        with open(local_file, 'rb') as f:
            train_labels = extract_labels(f, one_hot=True)

        start = time.clock()
        if isTest:
            print 'Creating MnistObject-dataset testset ...'
            testPath = os.path.join(self.config.datasetDir,
                                    self.config.teTrName[0])
            self.testset.createDataset(images=test_images,
                                       labels=test_labels,
                                       dstPath=testPath,
                                       imageSize=imageSize,
                                       scaleRange=scaleRange,
                                       isNoise=isNoise,
                                       noiseNum=noiseNum,
                                       noiseSize=noiseSize)
        if isTrain:
            print 'Creating MnistObject-dataset trainset ...'
            trainPath = os.path.join(self.config.datasetDir,
                                     self.config.teTrName[1])
            self.trainset.createDataset(images=train_images,
                                        labels=train_labels,
                                        dstPath=trainPath,
                                        imageSize=imageSize,
                                        scaleRange=scaleRange,
                                        isNoise=isNoise,
                                        noiseNum=noiseNum,
                                        noiseSize=noiseSize)
        end = time.clock()
        print 'Create MnistObject-dataset completely, cost time %f seconds' \
              % (end - start)


# The main function of the demo
def main():
    print sys.argv

    mnistDir = '/home/jielyu/Workspace/Python/AttentionModel/data/MNIST_data'
    dataDir = '/home/jielyu/Database/Mnist-dataset'
    config = Config(mnistDataDir=mnistDir,
                    datasetDir=dataDir)

    mnistObjectDataset = MnistObjectDataset(config)
    # mnistObjectDataset.createDataset(scaleRange=[1, 1])
    # mnistObjectDataset.createDataset(scaleRange=[0.3, 1.5])
    # raise ValueError('Break')
    mnistObjectDataset.readDataset()
    print mnistObjectDataset.trainset.images[0, :]

    # images, bbox = mnistObjectDataset.testset.getNextBatch()
    # print images.shape
    # print bbox.shape
    #
    # n = images.shape[0]
    # for i in range(0, n):
    #     plt.clf()
    #     plt.imshow(images[i, :, :, 0])
    #     plt.show()

    num = mnistObjectDataset.testset.getSampleNum()
    for i in range(0, num):
        # mnistObjectDataset.testset.showBbox(i)
        mnistObjectDataset.trainset.showBbox(i)

# The entry of the demo
if __name__ == '__main__':
    main()
