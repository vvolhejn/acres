# vim: sta:et:sw=2:ts=2:sts=2
# Written by Antonio Loquercio

import numpy as np
import cv2
import tensorflow as tf
from acres.binarization.smoother import Smoother

# Basic model parameters.
tf.app.flags.DEFINE_string('image_path', '../data/muenster/images/0282925037198-01_N95-2592x1944_scaledTo800x600bilinear.jpg',
                           """Path to the image to blur.""")
FLAGS = tf.app.flags.FLAGS

# Basic Constants


def smooth(SIGMA, FILTER_SIZE):

    Image_Placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    smoother = Smoother({'data': Image_Placeholder}, FILTER_SIZE, SIGMA)
    smoothed_image = smoother.get_output()

    init = tf.initialize_all_variables()

    with tf.Session() as sess:
        sess.run(init)

        image = cv2.imread(FLAGS.image_path) / 255.
        image = image.reshape((1, image.shape[0], image.shape[1], 3))

        # something something session
        kernel = _gauss_kernel(11, 10.)
        blurred_image = tf.nn.depthwise_conv2d([image], kernel, [1, 1, 1, 1],
                                               padding="SAME")


def main(argv=None):
    for sigma in np.linspace(0.0, 5.0, 21):
        filter_size = 21
        # for filter_size in range(1, 11, 2):
        # sigma = 10
        smooth(sigma, filter_size)

if __name__ == '__main__':
    tf.app.run()


def _normal(x, sigma=1):
    return np.exp(-x**2 / (2 * sigma**2)) / np.sqrt(2 * np.pi * sigma**2)


def _gauss_kernel(kernel_size=11, sigma=1):
    """
    Create a convolution kernel used for a gaussian blur.
    With kernel_size=2, sigma should be <=2, otherwise the curve is severely cropped
    """
    if sigma <= 0:
        # Unit vector with 1 in the middle
        kern1d = np.eye(1, kernel_size, kernel_size // 2)
    else:
        interval = kernel_size - 1
        x = np.linspace(-interval / 2., interval / 2., kernel_size)
        kern1d = _normal(x, sigma)

    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()  # Ensure the values sum up to 1
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernel_size, kernel_size, 1, 1))
    return out_filter
