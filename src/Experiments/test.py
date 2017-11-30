# encoing: utf8

import os
import sys
import numpy as np
import tensorflow as tf


def main():
    print sys.argv

    with tf.device('/gpu:1'):
        a = tf.constant([1,2,3,4,5,6], shape=[2, 3], name='a')
        b = tf.constant([2,3,4,5,6,7], shape=[3, 2], name='b')
        c = tf.matmul(a, b)

    sessConf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    sess = tf.Session(config=sessConf)
    print sess.run(c)

if __name__ == '__main__':
    main()

