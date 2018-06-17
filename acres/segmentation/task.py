import argparse
import os
import datetime
import re

from .model import model_fn
from . import dataset

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam


def run_experiment(hparams):
    model_name = "{}-{}-{}".format(
        os.path.basename(__file__),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
                  for key, value in sorted(hparams.values().items())))
    )

    # Create logdir name
    logdir = os.path.join(hparams.job_dir, model_name)
    # if not os.path.exists(hparams.job_dir):
    #     os.mkdir(hparams.job_dir)  # TF 1.6 will do this by itself

    # Construct the model
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=logdir,
        config=tf.estimator.RunConfig(tf_random_seed=42,
                                      session_config=tf.ConfigProto(inter_op_parallelism_threads=hparams.threads,
                                                                    intra_op_parallelism_threads=hparams.threads)),
        params={
            "optimizer": tf.train.AdamOptimizer,
            "learning_rate": 0.001,
        })

    train, dev, test = dataset.make_datasets(hparams.dataset_dir)

    # Train
    # with tqdm.trange(hparams.num_epochs) as t:
    #     for i in t:
    #         # steps argument + repeat?
    #         model.train(input_fn=lambda: train.make_one_shot_iterator().get_next())

    #         values = model.evaluate(input_fn=lambda: dev.make_one_shot_iterator().get_next(), name="dev")
    #         t.set_description("acc: {:.3f}".format(values["accuracy"]))

    # return model.predict(input_fn=lambda: test.make_one_shot_iterator().get_next())

    """Run the training and evaluate using the high level API"""

    # train_input = lambda: model.input_fn(
    #     hparams.train_files,
    #     num_epochs=hparams.num_epochs,
    #     batch_size=hparams.train_batch_size
    # )

    # # Don't shuffle evaluation data
    # eval_input = lambda: model.input_fn(
    #     hparams.eval_files,
    #     batch_size=hparams.eval_batch_size,
    #     shuffle=False
    # )

    train_spec = tf.estimator.TrainSpec(lambda: train.make_one_shot_iterator().get_next(),
                                        max_steps=hparams.train_steps,
                                        )
    # exporter = tf.estimator.FinalExporter('census',
    #                                       model.SERVING_FUNCTIONS[hparams.export_format])
    eval_spec = tf.estimator.EvalSpec(lambda: dev.make_one_shot_iterator().get_next(),
                                      steps=hparams.eval_steps,
                                      # exporters=[exporter],
                                      name="dev",
                                      )

    run_config = tf.estimator.RunConfig(model_dir=logdir,
                                        save_checkpoints_secs=600,
                                        )

    print('model dir {}'.format(run_config.model_dir))

    tf.estimator.train_and_evaluate(model,
                                    train_spec,
                                    eval_spec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Arguments
    # parser.add_argument(
    #     '--train-files',
    #     help='GCS or local paths to training data',
    #     nargs='+',
    #     required=True
    # )
    parser.add_argument(
        '--num-epochs',
        help="""\
      Maximum number of training data epochs on which to train.
      If both --max-steps and --num-epochs are specified,
      the training job will run for --max-steps or --num-epochs,
      whichever occurs first. If unspecified will run for --max-steps.\
      """,
        type=int,
    )
    parser.add_argument(
        '--train-batch-size',
        help='Batch size for training steps',
        type=int,
        default=40
    )
    parser.add_argument(
        '--eval-batch-size',
        help='Batch size for evaluation steps',
        type=int,
        default=40
    )
    # parser.add_argument(
    #     '--eval-files',
    #     help='GCS or local paths to evaluation data',
    #     nargs='+',
    #     required=True
    # )
    # Training arguments
    parser.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )

    # Argument to turn on all logging
    parser.add_argument(
        '--verbosity',
        choices=[
            'DEBUG',
            'ERROR',
            'FATAL',
            'INFO',
            'WARN'
        ],
        default='INFO',
    )
    # Experiment arguments
    parser.add_argument(
        '--train-steps',
        help="""\
      Steps to run the training job for. If --num-epochs is not specified,
      this must be. Otherwise the training job will run indefinitely.\
      """,
        type=int
    )
    parser.add_argument(
        '--eval-steps',
        help='Number of steps to run evalution for at each checkpoint',
        default=1,
        type=int
    )
    parser.add_argument(
        '--export-format',
        help='The input format of the exported SavedModel binary',
        choices=['JSON', 'CSV', 'EXAMPLE'],
        default='JSON'
    )

    parser.add_argument("--dataset-dir", default="data/", type=str, help="Dataset directory.")
    parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
    # parser.add_argument("--name", type=str)

    args = parser.parse_args()

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    hparams = hparam.HParams(**args.__dict__)
    run_experiment(hparams)
