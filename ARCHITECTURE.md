# acres architecture

## High-level problem
acres' end goal is to have a tool which receives an image containing a 1D barcode
(such as [EAN-13](https://en.wikipedia.org/wiki/International_Article_Number)) and produces some output image which is easier to decode than the original.

In the actual implementation we produce a mask which segments the input image, meaning it places each pixel into one of three classes:

- pixel not in a barcode
- pixel of a white bar
- pixel of a black bar

We assume the barcode is placed horizontally (the bars are vertical). Using this assumption, we may sample horizontal strips of the original image and try to sharpen these strips. If the sharpening works optimally and the barcode appears on at least one of these strips, it will be decoded. This is beneficial for performance, because it means it is not necessary to sharpen the entire image.

## Low-level problem
The network's input is a horizontal strip of the image. In practice, the whole image is 800x600 and the strip is 800x5. The output is a 796x3 prediction matrix, which contains the predicted classes of the central row of the input strip, except for the 2 leftmost and rightmost pixels.

If `I` is the input image (`I[x][y]` is the pixel in the zero-indexed `x`-th column and `y`-th row) and
`O` is the prediction matrix, then `O[x][c]` is the predicted probability that
`I[x+2][2]` has class `c`.

For each input image, we have a ground truth output mask telling us what the predictions ought to be. We use a standard cross-entropy loss function. In practice, the network predicts logits of the classes, which are then normalized using softmax, meaning we use Tensorflow's [softmax cross-entropy loss](https://www.tensorflow.org/api_docs/python/tf/nn/sparse_softmax_cross_entropy_with_logits).

Additionally, we add another term to the loss, dubbed the _change loss_. We want to punish the network for changing its predictions often, because mispredicting one pixel could make decoding the barcode impossible. Let `P` be the quantized prediction matrix, where `P[x]` is the index of the class with the highest probability for the `x`-th column. The change loss is then equal to (in NumPy notation)
```
np.mean((P[1:] != P[:-1]).astype(int)) * change_penalty
```
for some constant `change_penalty`. In words, it is equal to the number of pixels where the prediction changes divided by the total number of pixels, multiplied by `change_penalty`.

## Data
The network was trained on the WWU Muenster Barcode Database [1], which contains about 1000 images. This dataset only contains barcode images, but we also need masks telling us ground truth about where the barcode is. For this, we use another dataset containing masks for a subset (about 600 images) of the Muenster dataset [2]. The masks in this dataset only contain information about which pixels are part of a barcode and which aren't. To discriminate between black and white pixels of the barcode, we binarize the barcode image using [adaptive gaussian thresholding](https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html) and add the extracted information to the masks. This gives us all three classes.

We also discard images with vertically-placed barcodes (meaning the bars are _horizontal_). To do this, we use a simple method based on a Sobel filter. See [the `is_vertical()` function](./scripts/make_binarized_masks.py) for details.

## Network architecture
Our architecture is inspired by [3]. There is a series of 5 convolutional layers. Each layer has a stride meaning the input size gets lower and lower. We then upscale the last three of the layers' outputs using a transposed convolution. This should give us information about the image at different scales. We use the sum of these upscaled predictions as further input and add two more convolutional layers; the ultimate one predicts the logits for each of the classes.

Please refer to Figure 3 of [3] for a more detailed explanation.
The relevant code is in [the function `strided32_network()`](./binarization/model.py).

## Results
We trained the network for 30k steps, which took 6 hours on [Google Cloud Platform's BASIC_GPU tier](https://cloud.google.com/ml-engine/docs/tensorflow/machine-types#scale_tiers). The network achieved **a test set accuracy of 98.8%**. This number, however, is not very telling, considering over 90% of the input pixels are not in a barcode at all. Because of this, we add another metric, called _binarization accuracy_. It is the accuracy if we ignore correctly predicted non-barcode pixels. (The name, admittedly, is a bit confusing.) The network achieves a binarization accuracy of **84.2% on the test set**.

## References
- [1] WWU Muenster Barcode Database (http://cvpr.uni-muenster.de/research/barcode)
- [2] Robust Angle Invariant 1D Barcode Detection
  _Alessandro Zamberletti, Ignazio Gallo and Simone Albertini_,
  _Proceedings of the 2nd Asian Conference on Pattern Recognition (ACPR), Okinawa, Japan, 2013_
  [(link to the dataset)](http://artelab.dista.uninsubria.it/downloads/datasets/barcode/hough_barcode_1d/hough_barcode_1d.html)
- [3] Fully Convolutional Networks for Semantic Segmentation
  _Jonathan Long, Evan Shelhamer and Trevor Darrell_ (https://arxiv.org/abs/1411.4038)
