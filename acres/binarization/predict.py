import argparse
import os
import datetime
import re

import tqdm
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

from .model import model_fn
from . import dataset


def run_experiment(hparams):
    image_size = [hparams.image_height, hparams.image_width]
    # model_name = get_model_name(hparams)
    # Create logdir name (if local, this must exist)
    model_dir = os.path.join(hparams.job_dir, hparams.model_name)

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
                                      session_config=tf.ConfigProto(
                                          inter_op_parallelism_threads=hparams.threads,
                                          intra_op_parallelism_threads=hparams.threads),
                                      save_checkpoints_secs=120),
        params={
            "initial_learning_rate": 0.001,
            "learning_rate_decay_steps": 15000,
            "image_size": image_size,
            "context": hparams.patch_context,
            "network_name": hparams.network_name,
            "change_penalty": hparams.change_penalty
        })

    dataset_dict = dataset.make_datasets(hparams.dataset_dir,
                                         image_size=image_size,
                                         batch_size=hparams.batch_size,
                                         context=hparams.patch_context,
                                         stride=hparams.patch_stride,
                                         )
    train, dev, test = dataset_dict["datasets"]
    _, _, test_image_names = dataset_dict["image_names"]

    predictions_generator = model.predict(input_fn=lambda: test.make_one_shot_iterator().get_next())
    # It is unknown whether this computation is correct when the numbers are not divisible
    predictions_per_image = hparams.image_height // (hparams.patch_context * 2 + hparams.patch_stride)

    os.makedirs(os.path.join(model_dir, "predictions"), exist_ok=True)

    for image_name in tqdm.tqdm(test_image_names):
        cur_predictions = []
        for _ in range(predictions_per_image):
            cur_predictions.append(next(predictions_generator))

        cur_predictions = np.array(cur_predictions)

        # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        # cv2.imshow('image', (cur_predictions * 127).astype(np.uint8))
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(model_dir, "predictions", image_name + ".png"), (cur_predictions * 127).astype(np.uint8))


def get_model_name(hparams):
    skipped = {"job_dir", "dataset_dir", "verbosity", "threads"}

    filtered_hparams = [(key, value)
                        for key, value in sorted(hparams.values().items())
                        if key not in skipped]

    model_name = "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in filtered_hparams))
    )
    return model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        "--network-name",
        help="Name of the network to use.",
        type=str,
        default="baseline"
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        type=int,
        default=50
    )
    parser.add_argument(
        "--patch-context",
        help="How many pixels of 'context' to add around each patch",
        type=int,
        default=2
    )
    parser.add_argument(
        "--patch-stride",
        help="Stride between patches",
        type=int,
        default=1
    )
    parser.add_argument(
        "--image-width",
        help="Width to which to resize images",
        type=int,
        default=800
    )
    parser.add_argument(
        "--image-height",
        help="Width to which to resize images",
        type=int,
        default=600
    )
    parser.add_argument(
        "--change-penalty",
        help="Weight of penalization for the prediction changing values",
        type=float,
        default=0.0
    )
    # Training arguments
    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True
    )
    parser.add_argument(
        "--model-name",
        help="Name of the model to predict with.",
        required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
        "--verbosity",
        choices=[
            "DEBUG",
            "ERROR",
            "FATAL",
            "INFO",
            "WARN"
        ],
        default="INFO",
    )

    parser.add_argument("--dataset-dir", default="data/", type=str, help="Dataset directory.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # parser.add_argument("--name", type=str)

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Log GPU availability
    print("GPU:", tf.test.gpu_device_name())

    # Run the training job
    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
