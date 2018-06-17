#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import tqdm

from . import dataset

NUM_CLASSES = 3


def model_fn(features, labels, mode, params):
    conv1 = tf.layers.conv2d(features, filters=8, kernel_size=5,
                             padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=5,
                             padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=5,
                             padding="valid", activation=tf.nn.relu)
    # conv3t = tf.layers.conv2d_transpose(conv3, filters=16, kernel_size=5, strides=2,
    #                                     padding="same", activation=tf.nn.relu)
    # conv2t = tf.layers.conv2d_transpose(conv3t, filters=8, kernel_size=5, strides=2,
    #                                     padding="same", activation=tf.nn.relu)

    logits = tf.layers.conv2d(conv3, filters=NUM_CLASSES, kernel_size=3, padding="same", activation=None)
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions)  # predictions

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions,
                                   name="accuracy_op")

    # iou = tf.metrics.mean_iou(labels=tf.cast(labels, tf.int32),
    #                           predictions=tf.cast(predictions, tf.int32),
    #                           num_classes=2,
    #                           name="iou_op")

    tf.summary.scalar("accuracy", accuracy[1])
    # tf.summary.scalar("iou", iou[1][1][1]) # TODO: This doesn't seem to work right. Why?

    flat_labels = tf.reshape(labels, [-1])
    flat_predictions = tf.reshape(predictions, [-1])
    tf.summary.image(
        "confusion",
        tf.reshape(tf.confusion_matrix(flat_labels,
                                       flat_predictions,
                                       num_classes=NUM_CLASSES,
                                       # zero out correct predictions
                                       weights=tf.not_equal(flat_labels, flat_predictions),
                                       dtype=tf.float32),
                   [1, NUM_CLASSES, NUM_CLASSES, 1]),
        max_outputs=10,
    )

    eval_metric_ops = {
        "accuracy": accuracy,
        # "iou": iou,
    }

    if mode == tf.estimator.ModeKeys.TRAIN:
        # predicted_mask = tf.cast(predictions[0:1], tf.float32) * 255
        # target_mask = labels[0:1]
        # masked_image = tf.concat(
        #     [features[0:1] * tf.cast(binary_labels[0:1], tf.float32), features[0:1], features[0:1]],
        #     axis=3)
        # image_prediction = tf.reshape(tf.concat(predictions, axis=0),
        #                               [1, -1, params.get("image_size")[1] - 2 * params.get("context"), 1])
        image_features = tf.reshape(tf.concat(features, axis=0),
                                    [1, -1, params.get("image_size")[1], 1])
        image_features = tf.cast(image_features, tf.float32)
        # print(image_prediction.get_shape())
        print(image_features.get_shape())
        # tf.summary.image("plot", image_prediction)
        tf.summary.image("plot", image_features)

        optimizer_fn = params.get("optimizer", None)
        optimizer = optimizer_fn(params.get("learning_rate", None))
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode, predictions, loss, train_op,
            eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.EVAL:
        # print(labels.get_shape())
        # print(predictions.get_shape())

        # tf.summary.image("plot", tf.concat([tf.image.grayscale_to_rgb(predicted_mask), masked_image], axis=1))

        return tf.estimator.EstimatorSpec(mode, predictions, loss,
                                          eval_metric_ops=eval_metric_ops)
