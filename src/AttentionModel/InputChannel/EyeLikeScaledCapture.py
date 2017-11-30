
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


class EyeLikeScaledCapture(object):

    def __init__(self, config=Config()):
        self.config = config

    def buildGraph(self, images, points, scales):

        # Limit scale range
        scaleRange = self.config.focusedLengthScaleRange
        scales = tf.clip_by_value(scales,
                                  clip_value_min=scaleRange[0],
                                  clip_value_max=scaleRange[1])
        # Create bbox
        batchSize = tf.shape(images)[0]
        # print batchSize
        HW = tf.reshape(tf.shape(images)[1:3], [1, 2])
        HW = tf.tile(HW, [batchSize, 1])
        cropSize = tf.constant(value=self.config.minScaleSize,
                               dtype=tf.float32,
                               shape=[1, 2])
        cropSize = tf.tile(cropSize, [batchSize, 1])
        cropSize = cropSize*2/tf.cast(HW, tf.float32)
        cropSize = cropSize * tf.tile(scales, [1, 2])
        luYX = points - cropSize/2
        rdYX = luYX + cropSize
        bbox = tf.concat(concat_dim=1, values=[luYX, rdYX])
        bbox = (bbox + 1)/2

        # bbox_ind
        box_ind = tf.range(batchSize)
        targetSize = tf.constant(value=self.config.targetSize,
                                 dtype=tf.int32,
                                 shape=[2])

        targetImages = tf.image.crop_and_resize(image=images,
                                                boxes=bbox,
                                                box_ind=box_ind,
                                                crop_size=targetSize)

        # print targetImages
        return targetImages


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
    imgs = imgs/255-0.5
    # Points
    pts = np.zeros(shape=[2, 2], dtype=np.float32)
    pts[0, 0] = -0.1
    pts[0, 1] = 0.1
    pts[1, 0] = -0.1
    pts[1, 1] = 0.1
    # Scales
    scls = np.zeros(shape=[2, 1], dtype=np.float32)
    scls[0, 0] = 1
    scls[1, 0] = 2

    # Define the interface of EyeLikeCapture
    # Images
    images = tf.placeholder(
        dtype=tf.float32,
        shape=[None, 480, 640, 3],
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

    # Create EyeLikeCapture object
    eyeLikeCapture = EyeLikeScaledCapture()

    # Build Eye-like Graph
    targetImages = eyeLikeCapture.buildGraph(
        images=images,
        points=points,
        scales=scales)

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
        print 'cropImgs.shape = ', cropImgs.shape

        # Save multi-scale images
        outputDir = '../../../output/EyeLikeCapture/'
        if not os.path.isdir(outputDir):
            os.mkdir(outputDir)
        imgList = []
        for i in range(0, len(cropImgs)):
            imgPath = outputDir + 'scale_' + str(i) + '.png'
            img = np.uint8((cropImgs[i, :] + 0.5)*255)
            skimage.io.imsave(imgPath,
                              np.uint8(img))
            imgList.append(img)

        # Restore the image to uint8
        img = np.uint8(imgList[1])

        # Show
        plt.imshow(img)
        plt.show()

if __name__ == '__main__':
    main()
