# encoding:utf-8


# System libraries
import os
import sys

# 3rd-part libraries
import numpy as np
from matplotlib import pyplot as plt
from scipy import misc
import skimage.io
import tensorflow as tf

# Self-define libraries
from src.tools.Utils import Utils

# Define and implement a classtf_graph/name_scope_2
# to configure some parameters of class EyeLikeCapture
class Config(object):

    def __init__(self,
                 isAddContext=True,
                 nScale=3,
                 scaleFactor=3.0,
                 minScaleSize=32,
                 targetSize=64):
        """
        Function: Initialize the configurations of EyeLikeCapture
        """

        # self.inputShape = (480, 640, 3)
        self.isAddContext = isAddContext
        # The number of scales
        self.nScale = nScale
        # The factor between two adjacent scales
        self.scaleFactor = scaleFactor
        # The size of minimum scale
        self.minScaleSize = minScaleSize
        # Unused, The range of size like focused length
        self.focusedLengthScaleRange = [0.25, 4]
        # The final size of images with all scales
        self.targetSize = targetSize
        # The size of all scales
        self.multiScaleSize = \
            [np.int32(self.minScaleSize*(self.scaleFactor**i))
             for i in range(0, self.nScale)]

    def computeMultiScaleSize(self):
        """
        Function: Re-generate the sizes of all scales from new parameters
        :return: None
        """

        self.multiScaleSize = \
            [np.int32(self.minScaleSize * (self.scaleFactor ** i))
             for i in range(0, self.nScale)]


# Define and implement a class to build a graph
# for capturing multi-scale images like eyes in tensorflow
class EyeLikeCapture(object):

    # Initialize parameters and members
    def __init__(self, config=Config()):
        """
        Function: Initialize members of EyeLikeCapture object
        :param config:
        """

        # Configuration
        self.config = config
        # Output operations
        # self.multiScaleImages = dict()

    # Used to build a graph to analog eye
    def buildGraph(self, images, points, name='RetinaLikeNet', reuse=True):
        """
        Function: Build a graph to extract a glimpses and resize
        :param images: 4D, [batchSize,h,w,channel], float32
        :param points: 2D, [batchSize, 2], float32, [h, w], [-1, 1]
        :return:
            dict{i:scaledImages}, 4D, [batchSize, hh, ww, channel], float32
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # Size of target images
            targetSize = tf.constant(
                value=self.config.targetSize,
                dtype=tf.int32,
                shape=[2])

            # Compute the sizes of all scales
            multiScaleImages = []
            for i in range(0, self.config.nScale):

                # Size of a scale
                size = tf.constant(
                    value=self.config.multiScaleSize[i],
                    dtype=tf.int32,
                    shape=[2])

                # Extract glimpses and resize
                if i == self.config.nScale-1:
                    if self.config.isAddContext:
                        # Add context scale
                        multiScaleImages.append(
                            tf.image.resize_images(images, targetSize))
                        break

                # Add local patch
                multiScaleImages.append(
                    tf.image.resize_images(
                        tf.image.extract_glimpse(
                            input=images,
                            size=size,
                            offsets=points),
                        targetSize))

            # if self.config.isAddContext:
            #     # Add context scale
            #     multiScaleImages.append(
            #         tf.image.resize_images(images, targetSize))
        # Return
        return multiScaleImages

    def getRectangle(self, point):
        """
        Function: Get rectangles of multi-scale images
        :param point: Point of glimpse, 2D, [numImages,2], [h, w], [-1, 1]
        :return:
            Rectangles of all scales, 2D, [numImages, 4], [luy,lux,rby, rbx]
        """
        pass

    def drawRectangle(self, image, point):
        """
        Function: Draw multi-scale rectangles on an image
        :param image: Images, 4D, [numImages, h, w, channel]
        :param point: Points, 2D, [numImages, 2], [h, w] [-1, 1]
        :return:
            Images with multi-scale rectangles, 4D
        """
        pass


# The main function of the demo
def main():
    print sys.argv

    # Read data
    # Images
    imgPath = \
        '/home/jielyu/Workspace/Python/AttentionModel/data/' \
        'test/v_0093_img_000011.jpg'
    imgPath = \
        '/home/jielyu/Workspace/Python/AttentionModel/data/' \
        'test/I_00000010.jpg'
    img = misc.imread(imgPath)
    imgs = np.zeros(shape=[2, img.shape[0], img.shape[1], img.shape[2]],
                    dtype=np.float32)
    imgs[0, :] = img
    imgs[1, :] = img
    imgs = imgs/255-0.5
    # Points
    pts = np.zeros(shape=[2, 2], dtype=np.float32)
    pts[0, 0] = 0.2
    pts[0, 1] = 0.2
    pts[1, 0] = 0.2
    pts[1, 1] = 0.2
    # Scales
    scls = np.zeros(shape=[2, 1], dtype=np.float32)
    scls[0, 0] = 1
    scls[1, 0] = 1

    # Define the interface of EyeLikeCapture
    # Images
    images = tf.placeholder(
        dtype=tf.float32,
        shape=[None, imgs.shape[1], imgs.shape[2], imgs.shape[3]],
        name='images')
    # Points
    points = tf.placeholder(
        dtype=tf.float32,
        shape=[None, 2],
        name='fixedPoints')
    # Scales
    scales = tf.placeholder(
        dtype=tf.float32,
        shape=[None, 1],
        name='focusedLengthScales')

    config = Config(minScaleSize=128,
                    nScale=3,
                    scaleFactor=2.4,
                    targetSize=128,
                    isAddContext=False)
    # Create EyeLikeCapture object
    eyeLikeCapture = EyeLikeCapture(config=config)

    # Build Eye-like Graph
    targetImages = eyeLikeCapture.buildGraph(
        images=images,
        points=points)

    # Create session
    sessConf = tf.ConfigProto()
    sessConf.gpu_options.allow_growth = True
    with tf.Session(config=sessConf) as sess:

        # Initialize all variables
        initOp = tf.initialize_all_variables()
        sess.run(initOp)

        # Run the graph
        feed_dict = {images: imgs, points: pts, scales: scls}
        cropImgs = sess.run(targetImages, feed_dict=feed_dict)

        # Save multi-scale images
        outputDir = '../../../output/EyeLikeCapture/'
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)
        imgList = []
        for i in range(0, len(cropImgs)):
            tImgPath = outputDir + 'scale_' + str(i) + '.png'
            img = np.uint8((cropImgs[i][0, :] + 0.5) * 255)
            skimage.io.imsave(tImgPath, img)
            imgList.append(img)

        size = config.multiScaleSize[-1]
        maxSize = [size, size, 3]
        print maxSize
        eyeLikeImg = np.zeros(shape=maxSize, dtype=np.uint8)
        for i in range(0, config.nScale):

            img = imgList[config.nScale-1-i]
            size = config.multiScaleSize[config.nScale-1-i]
            size = [size, size]
            t_img = misc.imresize(img, size)
            minH = maxSize[0]/2 - size[0]/2
            minW = maxSize[1]/2 - size[1]/2
            maxH = minH + size[0]
            maxW = minW + size[1]
            eyeLikeImg[minH:maxH, minW:maxW, :] = t_img
        plt.imshow(eyeLikeImg)
        for i in range(0, config.nScale):
            size = config.multiScaleSize[config.nScale - 1 - i]
            size = [size, size]
            minH = maxSize[0] / 2 - size[0] / 2
            minW = maxSize[1] / 2 - size[1] / 2
            maxH = minH + size[0]
            maxW = minW + size[1]
            Utils.drawRect(minW, minH, maxW, maxH, 'yellow')
            plt.axis('off')
        foo_fig = plt.gcf()  # 'get current figure
        tImgPath = outputDir + 'concat' + '.png'
        foo_fig.savefig(tImgPath, format='png', dpi=128, bbox_inches='tight')

        plt.clf()
        img = misc.imread(imgPath)
        plt.imshow(img)
        for i in range(0, config.nScale):
            size = config.multiScaleSize[config.nScale - 1 - i]
            size = [size, size]
            minH = img.shape[0] / 2 * (1 + pts[0, 1]) - size[0] / 2
            minW = img.shape[1] / 2 * (1 + pts[0, 0]) - size[1] / 2
            maxH = minH + size[0]
            maxW = minW + size[1]
            Utils.drawRect(minW, minH, maxW, maxH, 'yellow')
            plt.axis('off')
        foo_fig = plt.gcf()  # 'get current figure
        tImgPath = outputDir + 'box' + '.png'
        plt.show()
        foo_fig.savefig(tImgPath, format='png', dpi=128, bbox_inches='tight')

        # # Restore the image to uint8
        # img = np.uint8((cropImgs[0][0, :] + 0.5)*255)
        #
        # # Show
        # plt.imshow(img)
        # plt.show()


# The entry of the demo
if __name__ == '__main__':
    main()
