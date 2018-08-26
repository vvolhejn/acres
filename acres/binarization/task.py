"""
The entry-point module for network training.
Based loosely on a GCP ML Engine example:
https://github.com/GoogleCloudPlatform/cloudml-samples/blob/master/census/customestimator/trainer/task.py
"""
import argparse
import os
import datetime
import re

from .model import model_fn, ALLOWED_NETWORK_NAMES
from . import dataset

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam

SEED = 592  # Random seed used for dataset splitting; used for consistency in predict.py


def run_experiment(hparams):
    image_size = [hparams.image_height, hparams.image_width]
    model_name = get_model_name(hparams)

    # Create logdir name (if local, this must exist)
    logdir = os.path.join(hparams.job_dir, model_name)

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=logdir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
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
                                         seed=592,
                                         )
    train, dev, test = dataset_dict["datasets"]

    # Run the training and evaluate using the Estimator API.
    train_spec = tf.estimator.TrainSpec(lambda: train.make_one_shot_iterator().get_next(),
                                        max_steps=hparams.train_steps,
                                        )

    eval_spec = tf.estimator.EvalSpec(lambda: dev.make_one_shot_iterator().get_next(),
                                      steps=hparams.eval_steps,
                                      name="dev",
                                      )

    # The results of training may be inspected using Tensorboard.
    tf.estimator.train_and_evaluate(model,
                                    train_spec,
                                    eval_spec)


def get_model_name(hparams):
    """
    Generate a (hopefully) unique name for the model based on the hyperparameters and datetime information.
    """
    skipped = {"job_dir", "dataset_dir", "verbosity", "threads"}

    filtered_hparams = [(key, value)
                        for key, value in sorted(hparams.values().items())
                        if key not in skipped]

    # Trick from https://github.com/ufal/npfl114/ - abbreviate the hyperparameters' names.
    hyperparams_str = ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                                for key, value in filtered_hparams))

    model_name = "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        hyperparams_str,
    )
    return model_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Datset arguments
    parser.add_argument(
        "--dataset-dir",
        help="Dataset directory.",
        default="data/",
        type=str,
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        type=int,
        default=40,
    )
    parser.add_argument(
        "--patch-context",
        help="How many pixels of 'context' to add around each patch; each patch then has 1 + 2*context rows.",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--patch-stride",
        help="Stride between patches; a stride of 1 uses all pixels with no overlap",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--image-width",
        help="Width to which to resize images",
        type=int,
        default=800,
    )
    parser.add_argument(
        "--image-height",
        help="Height to which to resize images",
        type=int,
        default=600,
    )
    # Network arguments
    parser.add_argument(
        "--network-name",
        help="Name of the network to use.",
        type=str,
        choices=ALLOWED_NETWORK_NAMES,
        default="strided32",
    )
    parser.add_argument(
        "--change-penalty",
        help="Weight of penalization for changing the prediction of neighboring pixels",
        type=float,
        default=0.0,
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
    # Experiment arguments
    parser.add_argument(
        "--job-dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--train-steps",
        help="Steps to run the training job for.",
        type=int,
    )
    parser.add_argument(
        "--eval-steps",
        help="Number of steps to run evalution for at each checkpoint",
        default=10,
        type=int,
    )

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
