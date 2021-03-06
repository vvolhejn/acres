#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

NUM_CLASSES = 3
ALLOWED_NETWORK_NAMES = ["baseline", "strided", "strided32"]


def model_fn(features, labels, mode, params, config):
    """
    A function representing our network. Used in the tf.Estimator API; follows the required signature.

    Shape of features:    (batch size, 1+2*context, image_width, 1)
    Shape of predictions: (batch size, image_width-2*context)
    """
    context = params["context"]  # Number of rows above and below target in features

    # Select the network function to use based on `params`.
    network_name = params["network_name"]
    if network_name not in ALLOWED_NETWORK_NAMES:
        raise ValueError("network_name must be one of: {}".format(ALLOWED_NETWORK_NAMES))
    network_dict = {
        "baseline": baseline_network,
        "strided": strided_network,
        "strided32": strided32_network,
    }
    network = network_dict[network_name]

    num_params = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()])
    print("Number of parameters: {}".format(num_params))

    # Perform prediction using the network function.
    logits = network(features, context, params.get("image_size")[1])
    predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    # If we are predicting, there are no labels and therefore we must return now.
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions_continuous = tf.nn.softmax(logits)
        return tf.estimator.EstimatorSpec(mode, predictions_continuous)

    # Calculate metrics for Tensorboard
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predictions,
                                   name="accuracy_op")
    # Binarization accuracy ignores correct predictions of background which inflate the accuracy
    binarization_accuracy_mask = 1.0 - (tf.cast(tf.equal(labels, 1), tf.float32) *
                                        tf.cast(tf.equal(predictions, 1), tf.float32))
    binarization_accuracy = tf.metrics.accuracy(labels=labels,
                                                predictions=predictions,
                                                weights=binarization_accuracy_mask,
                                                name="binarization_accuracy_op")
    eval_metric_ops = {
        "accuracy": accuracy,
        "binarization_accuracy": binarization_accuracy,
    }

    cross_entropy_loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)
    # Punish the network for changing its predictions between consecutive pixels.
    change_loss = tf.reduce_mean(tf.cast(tf.not_equal(predictions[1:], predictions[:-1]), tf.float32))
    loss = cross_entropy_loss + params.get("change_penalty") * change_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        # Ensure the metrics are evaluated on the training set as well.
        for name, (metric_value, update_op) in eval_metric_ops.items():
            tf.summary.scalar(name, update_op)

        # Decay the learning rate.
        learning_rate = tf.train.exponential_decay(
            learning_rate=params["initial_learning_rate"],
            global_step=tf.train.get_global_step(),
            decay_steps=params["learning_rate_decay_steps"],
            decay_rate=0.1
        )

        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(mode, predictions, loss, train_op,
                                          eval_metric_ops=eval_metric_ops)

    elif mode == tf.estimator.ModeKeys.EVAL:
        # When evaluating, produce images to visualise the predictions. This allows us to see the network's
        # performance during training in an intuitive way.
        image_features = tf.reshape(tf.concat(features, axis=0),
                                    [1, -1, params.get("image_size")[1], 1])
        image_w = params.get("image_size")[1] - 2 * context
        image_h = tf.shape(image_features)[1]

        image_prediction = tf.reshape(tf.concat(predictions, axis=0), [1, -1, image_w, 1])
        image_prediction = tf.cast(image_prediction, tf.float32)

        tf.summary.image("prediction", image_prediction)

        image_prediction = tf.image.resize_images(image_prediction, [image_h, image_w])

        image_features = tf.image.crop_to_bounding_box(image_features, 0, context, image_h, image_w)

        image_features_colored = tf.image.grayscale_to_rgb(image_features[0:1])
        image_prediction_colored = 255 * tf.concat([(image_prediction[0:1]),
                                                    (2. - image_prediction[0:1]),
                                                    (2. - image_prediction[0:1])],
                                                   axis=3)

        MASK_WEIGHT = 0.25
        masked_image = (MASK_WEIGHT * image_prediction_colored +
                        (1 - MASK_WEIGHT) * image_features_colored)
        tf.summary.image("masked", masked_image)

        # Hack to make TF produce images in eval mode as well.
        eval_summary_hook = tf.train.SummarySaverHook(
            save_steps=1,
            output_dir=config.model_dir + "/eval_dev",
            summary_op=tf.summary.merge_all())

        return tf.estimator.EstimatorSpec(mode, predictions, loss,
                                          eval_metric_ops=eval_metric_ops,
                                          evaluation_hooks=[eval_summary_hook])


def baseline_network(features, context, width):
    # Number of parameters: 16548
    conv1 = tf.layers.conv2d(features, filters=8, kernel_size=5,
                             padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv1, filters=16, kernel_size=5,
                             padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, filters=32, kernel_size=context * 2 + 1,
                             padding="valid", activation=tf.nn.relu)
    conv3_flat = tf.squeeze(conv3, axis=[1])
    logits = tf.layers.conv1d(conv3_flat, filters=NUM_CLASSES, kernel_size=3, padding="same", activation=None)
    return logits


def strided_network(features, context, width):
    preconv1 = tf.layers.conv2d(features, filters=16, kernel_size=5,
                                padding="same", activation=tf.nn.relu)
    preconv2 = tf.layers.conv2d(preconv1, filters=16, kernel_size=(context * 2 + 1),
                                padding="valid", activation=tf.nn.relu)

    downscale1 = tf.layers.conv2d(preconv2, filters=32, kernel_size=5, strides=2,
                                  padding="same", activation=tf.nn.relu)
    downscale2 = tf.layers.conv2d(downscale1, filters=64, kernel_size=5, strides=2,
                                  padding="same", activation=tf.nn.relu)

    upscale1 = tf.layers.conv2d_transpose(downscale2, filters=32, kernel_size=5, strides=[1, 2],
                                          padding="same", activation=tf.nn.relu)
    upscale2 = tf.layers.conv2d_transpose(upscale1, filters=16, kernel_size=5, strides=[1, 2],
                                          padding="same", activation=tf.nn.relu)
    logits = tf.layers.conv2d(upscale2 + preconv2,
                              filters=NUM_CLASSES, kernel_size=3, padding="same", activation=None)
    logits = tf.squeeze(logits, axis=[1])
    return logits


def strided32_network(features, context, width):
    # The deepest network.
    # This network assumes the image width is divisible by 32 - we use 800px in practice.

    preconv1 = tf.layers.conv2d(features, filters=16, kernel_size=5,
                                padding="same", activation=tf.nn.relu)
    preconv2 = tf.layers.conv2d(preconv1, filters=32, kernel_size=(context * 2 + 1),
                                padding="valid", activation=tf.nn.relu)

    downscale1 = tf.layers.conv2d(preconv2, filters=32, kernel_size=[1, 5], strides=2,
                                  padding="same", activation=tf.nn.relu)
    downscale2 = tf.layers.conv2d(downscale1, filters=32, kernel_size=[1, 5], strides=2,
                                  padding="same", activation=tf.nn.relu)
    downscale3 = tf.layers.conv2d(downscale2, filters=32, kernel_size=[1, 5], strides=2,
                                  padding="same", activation=tf.nn.relu)
    downscale4 = tf.layers.conv2d(downscale3, filters=32, kernel_size=[1, 5], strides=2,
                                  padding="same", activation=tf.nn.relu)
    downscale5 = tf.layers.conv2d(downscale4, filters=32, kernel_size=[1, 5], strides=2,
                                  padding="same", activation=tf.nn.relu)
    # The most downscaled layer, `downscaled5`, has a width of only 1/32 of the original image.

    # Upscale the deeper layers back to the target width.
    upscale5 = tf.layers.conv2d_transpose(
        downscale5, filters=32, kernel_size=[1, 5], strides=[1, 32],
        padding="same", activation=tf.nn.relu
    )[:, :, context:width - context, :]
    upscale4 = tf.layers.conv2d_transpose(
        downscale4, filters=32, kernel_size=[1, 5], strides=[1, 16],
        padding="same", activation=tf.nn.relu
    )[:, :, context:width - context, :]
    upscale3 = tf.layers.conv2d_transpose(
        downscale3, filters=32, kernel_size=[1, 5], strides=[1, 8],
        padding="same", activation=tf.nn.relu
    )[:, :, context:width - context, :]

    # Combining the masks of various granularity should give the network information about
    # multiple scales, enabling it to make precise predictions.
    postconv1 = tf.layers.conv2d(preconv2 + upscale5 + upscale4 + upscale3,
                                 filters=32, kernel_size=[1, 5], padding="same", activation=tf.nn.relu)
    logits = tf.layers.conv2d(postconv1,
                              filters=NUM_CLASSES, kernel_size=5, padding="same", activation=None)
    logits = tf.squeeze(logits, axis=[1])
    return logits
