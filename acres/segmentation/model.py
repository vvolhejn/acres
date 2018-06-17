#!/usr/bin/env python3
import argparse
import datetime
import os
import re

import numpy as np
import tensorflow as tf
import tqdm

from . import dataset


def model_fn(features, labels, mode, params):
    conv1 = tf.layers.conv2d(features, filters=8, kernel_size=5,
                             padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=5, strides=2,
                             padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=5, strides=2,
                             padding="same", activation=tf.nn.relu)
    conv3t = tf.layers.conv2d_transpose(conv3, filters=16, kernel_size=5, strides=2,
                                        padding="same", activation=tf.nn.relu)
    conv2t = tf.layers.conv2d_transpose(conv3t, filters=8, kernel_size=5, strides=2,
                                        padding="same", activation=tf.nn.relu)

    logits = tf.layers.conv2d(conv2t, filters=1, kernel_size=3, padding="same", activation=None)
    predictions = tf.greater_equal(logits, 0)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions)  # predictions

    binary_labels = tf.greater(labels, 0)
    loss = tf.losses.sigmoid_cross_entropy(binary_labels, logits)

    accuracy = tf.metrics.accuracy(labels=binary_labels,
                                   predictions=predictions,
                                   name="accuracy_op")

    iou = tf.metrics.mean_iou(labels=tf.cast(binary_labels, tf.int32),
                              predictions=tf.cast(predictions, tf.int32),
                              num_classes=2,
                              name="iou_op")

    tf.summary.scalar("accuracy", accuracy[1])
    # tf.summary.scalar("iou", iou[1][1][1]) # TODO: This doesn't seem to work right. Why?

    eval_metric_ops = {
        "accuracy": accuracy,
        "iou": iou,
    }

    if mode == tf.estimator.ModeKeys.TRAIN:
        predicted_mask = tf.cast(predictions[0:1], tf.float32) * 255
        target_mask = labels[0:1]
        masked_image = tf.concat(
            [features[0:1] * tf.cast(binary_labels[0:1], tf.float32), features[0:1], features[0:1]],
            axis=3)

        tf.summary.image("plot", tf.concat([tf.image.grayscale_to_rgb(predicted_mask), masked_image], axis=1))

        optimizer_fn = params.get("optimizer", None)
        optimizer = optimizer_fn(params.get("learning_rate", None))
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode, predictions, loss, train_op,
            eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, predictions, loss,
                                          eval_metric_ops=eval_metric_ops)


def main():
    # Fix random seed
    np.random.seed(42)
    tf.set_random_seed(42)

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/", type=str, help="Batch size.")
    parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=3, type=int, help="Number of epochs.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    parser.add_argument("--name", type=str)
    args = parser.parse_args()

    # Create logdir name
    args.logdir = "logs/{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
    )
    # if not os.path.exists("logs"):
    #     os.mkdir("logs")  # TF 1.6 will do this by itself

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.logdir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=args.threads,
                                                                    intra_op_parallelism_threads=args.threads)),
        params={
            "optimizer": tf.train.AdamOptimizer,
            "learning_rate": 0.001,
        })

    train, dev, test = dataset.make_datasets(args.dataset_dir)

    # Train
    with tqdm.trange(args.epochs) as t:
        for i in t:
            # steps argument + repeat?
            model.train(input_fn=lambda: train.make_one_shot_iterator().get_next())

            values = model.evaluate(input_fn=lambda: dev.make_one_shot_iterator().get_next(), name="dev")
            t.set_description("acc: {:.3f}".format(values["accuracy"]))

    return model.predict(input_fn=lambda: test.make_one_shot_iterator().get_next())


if __name__ == "__main__":
    main()

# l = list(main())
# import cv2
# cv2.namedWindow("x", cv2.WINDOW_NORMAL)
# cv2.imshow("x", 1.0 * l[0])
# cv2.waitKey()
