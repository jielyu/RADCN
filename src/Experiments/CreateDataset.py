# encoding: utf8

import os
import sys

from src.Dataset.MnistObjectDataset import Config as MnistConfig
from src.Dataset.MnistObjectDataset import MnistObjectDataset

def createMnistDataset():
    mnistDir = os.path.join('../..', 'data/MNIST_data')
    dataDir = os.path.join('/home/jack', 'Database/Test/Mnist-dataset')
    config = MnistConfig(mnistDataDir=mnistDir,
                         datasetDir=dataDir)
    dataset = MnistObjectDataset(config=config)
    dataset.createDataset(imageSize=[28, 28])
    dataset.readDataset()
    print 'tr_img_shape = ', dataset.trainset.images.shape
    print 'tr_bbox_shape = ', dataset.trainset.bbox.shape
    print 'te_img_shape = ', dataset.testset.images.shape
    print 'te_bbox_shape = ', dataset.testset.bbox.shape
    dataset.trainset.showBbox(0)


def createMnistObjectDataset():
    mnistDir = os.path.join('../..', 'data/MNIST_data')
    dataDir = os.path.join('/home/jack', 'Database/Test/MnistObject-dataset')
    config = MnistConfig(mnistDataDir=mnistDir,
                         datasetDir=dataDir)
    dataset = MnistObjectDataset(config=config)
    dataset.createDataset(imageSize=[56, 56])
    dataset.readDataset()
    print 'tr_img_shape = ', dataset.trainset.images.shape
    print 'tr_bbox_shape = ', dataset.trainset.bbox.shape
    print 'te_img_shape = ', dataset.testset.images.shape
    print 'te_bbox_shape = ', dataset.testset.bbox.shape
    dataset.trainset.showBbox(0)


def createMnistScaledObjectDataset():
    mnistDir = os.path.join('../..', 'data/MNIST_data')
    dataDir = os.path.join('/home/jack', 'Database/Test/MnistScaledObject-dataset')
    config = MnistConfig(mnistDataDir=mnistDir,
                         datasetDir=dataDir)
    dataset = MnistObjectDataset(config=config)
    dataset.createDataset(imageSize=[56, 56], scaleRange=[0.3, 1.5])
    dataset.readDataset()
    print 'tr_img_shape = ', dataset.trainset.images.shape
    print 'tr_bbox_shape = ', dataset.trainset.bbox.shape
    print 'te_img_shape = ', dataset.testset.images.shape
    print 'te_bbox_shape = ', dataset.testset.bbox.shape
    dataset.trainset.showBbox(0)


def createMnistScaledNoisedObjectDataset():
    mnistDir = os.path.join('../..', 'data/MNIST_data')
    dataDir = os.path.join('/home/jack', 'Database/Test/MnistScaledNoisedObject-dataset')
    config = MnistConfig(mnistDataDir=mnistDir,
                         datasetDir=dataDir)
    dataset = MnistObjectDataset(config=config)
    dataset.createDataset(imageSize=[56, 56],
                          scaleRange=[0.3, 1.5],
                          isNoise=True,
                          noiseNum=6,
                          noiseSize=[6, 6])
    dataset.readDataset()
    print 'tr_img_shape = ', dataset.trainset.images.shape
    print 'tr_bbox_shape = ', dataset.trainset.bbox.shape
    print 'te_img_shape = ', dataset.testset.images.shape
    print 'te_bbox_shape = ', dataset.testset.bbox.shape
    dataset.trainset.showBbox(0)


def createClutteredTranslatedMnist100x100():
    mnistDir = os.path.join('../..', 'data/MNIST_data')
    dataDir = os.path.join('/home/jack', 'Database/Test/ClutteredTranslatedMnist100x100-dataset')
    config = MnistConfig(mnistDataDir=mnistDir,
                         datasetDir=dataDir)
    dataset = MnistObjectDataset(config=config)
    dataset.createDataset(imageSize=[100, 100],
                          # scaleRange=[, 1.5],
                          isNoise=True,
                          noiseNum=8,
                          noiseSize=[8, 8])


def main():
    print sys.argv
    createMnistDataset()
    createMnistObjectDataset()
    createMnistScaledObjectDataset()
    createMnistScaledNoisedObjectDataset()
    createClutteredTranslatedMnist100x100()

if __name__ == '__main__':
    main()

