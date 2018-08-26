# acres
acres is a tool for sharpening barcodes. Specifically, it is an implementation
of a convolutional neural network designed for preprocessing blurry images of
1D barcodes.

For details of the architecture and implementation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## Example
The network takes an image and performs segmentation; for every pixel, it tries to predict whether it is:

 - not a part of a barcode
 - a white bar
 - a black bar

### Input image
![Input image](./doc/33_4009418180416-01_N95-2592x1944_scaledTo800x600bilinear.jpg)
### Prediction
![Prediction](./doc/33_4009418180416-01_N95-2592x1944_scaledTo800x600bilinear_prediction.png)
### Ground truth - what the network would output in the ideal case
![Ground truth](./doc/33_4009418180416-01_N95-2592x1944_scaledTo800x600bilinear_ground_truth.png)

Even though the image is heavily blurred, the network manages to extract a reasonable prediction.

## Installation
acres requires Python 3.4 or higher. We recommend to use virtualenv for
installation; the `install.sh` script should install the package in this way.
Certain scripts in the `scripts` directory also require OpenCV (which must be
installed separately), although this is not necessary for training.

## Usage

### Local training
To run training locally, we may use:
```
python -m acres.binarization.task \
    --dataset-dir data/muenster_blur/ \
    --train-steps 500 \
    --job-dir logs/example/
```
where `--dataset-dir` is the path to the dataset (should have `images` and `masks` subdirectories)
and `--job-dir` is the directory which will contain log files. The training should finish in a few
minutes on a regular laptop.

To see the progress of the training, we can use Tensorboard, which should be installed automatically with Tensorflow.
Run `tensorboard --logdir logs/example/ --port 6006` and open `localhost:6006` in a browser.
- The _Scalars_ tab contains charts of loss, accuracy and binarization accuracy.
- The _Images_ tab contains visualisations of predictions during training. Under `masked`, there are images
  overlaid with the predicted classes - one is highlighted in red, the other in blue.

### Training on Google Cloud Platform
Because acres uses Tensorflow's Estimator API, it can be seamlessly deployed on Google Cloud Platform's
[ML Engine](https://cloud.google.com/ml-engine/). We can also use the GPU for computations, which
gives us a great speed-up.

For example, this is the command (except for different GCP Storage bucket names) which was used
to train the final network:
```
gcloud ml-engine jobs submit training example_acres_training_(date "+%Y_%m_%d_%H%M%S") \
    --module-name acres.binarization.task \
    --package-path acres/ \
    --job-dir gs://my-bucket-for-logs/change_penalty/ \
    --config gcp_config.yaml  \
    --region us-east1 \
    --scale-tier BASIC_GPU \
-- \
    --network-name strided32 \
    --dataset-dir gs://my-bucket-for-data/muenster_blur \
    --batch-size 500 \
    --train-steps 30000 \
    --change-penalty 100
```

### Predictions
