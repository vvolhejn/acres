"""
A script for debugging the acres.binarization.dataset module.
Displays the images produced by the dataset, plus some statistics.
"""
import tensorflow as tf
import cv2
import numpy as np

import acres.binarization.dataset as dataset

x = dataset.make_datasets("data/muenster_blur", [300, 400], batch_size=50, context=2)[2]
sess = tf.Session()
next_element = x.make_one_shot_iterator().get_next()

while True:
    x, y = sess.run(next_element)
    print(x.shape, y.shape)
    print(np.min(y), np.max(y))
    print(np.sum(x))

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow('image', (y * 127).astype(np.uint8))
    cv2.waitKey(0)
