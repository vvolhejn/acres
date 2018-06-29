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


def model_fn(features, labels, mode, params, config):
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

    # Ignore correct predictions of background which inflate the accuracy
    binarization_accuracy_mask = 1.0 - (tf.cast(tf.equal(labels, 0), tf.float32) *
                                        tf.cast(tf.equal(predictions, 0), tf.float32))

    binarization_accuracy = tf.metrics.accuracy(labels=labels,
                                                predictions=predictions,
                                                weights=binarization_accuracy_mask,
                                                name="binarization_accuracy_op")

    # iou = tf.metrics.mean_iou(labels=tf.cast(labels, tf.int32),
    #                           predictions=tf.cast(predictions, tf.int32),
    #                           num_classes=2,
    #                           name="iou_op")

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
        "binarization_accuracy": binarization_accuracy,
        # "iou": iou,
    }

    if mode == tf.estimator.ModeKeys.TRAIN:
        for name, (metric_value, update_op) in eval_metric_ops.items():
            tf.summary.scalar(name, update_op)

        optimizer_fn = params.get("optimizer", None)
        optimizer = optimizer_fn(params.get("learning_rate", None))
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.EVAL:
        # image_features = tf.concat(features, axis=0)
        image_features = tf.reshape(tf.concat(features, axis=0),
                                    [1, -1, params.get("image_size")[1], 1])
        context = params.get("context")
        image_w = params.get("image_size")[1] - 2 * context
        image_h = tf.shape(image_features)[1]

        image_features = tf.cast(image_features, tf.float32)
        tf.summary.image("features", image_features)

        image_prediction = tf.reshape(tf.concat(predictions, axis=0),
                                      [1, -1, image_w, 1])
        image_prediction = tf.cast(image_prediction, tf.float32)

        tf.summary.image("prediction", image_prediction)

        image_prediction = tf.image.resize_images(image_prediction, [image_h, image_w])

        image_features = tf.image.crop_to_bounding_box(image_features, 0, context, image_h, image_w)
        masked_image = tf.concat(
            [image_prediction[0:1] * 100,
             image_features[0:1],
             image_features[0:1]],
            axis=3)

        print(image_features.get_shape())
        print(image_prediction.get_shape())
        print(masked_image.get_shape())

        tf.summary.image("masked", masked_image)

        # Hack to make TF produce images in eval mode as well.
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=config.model_dir + "/eval_dev",
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode, predictions, loss,
                                          eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[eval_summary_hook])
