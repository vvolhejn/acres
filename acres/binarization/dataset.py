"""
Functions used for constructing a dataset of images and ground truth masks.
"""
import os
import random

import tensorflow as tf


def make_datasets(dataset_dir, image_size, batch_size, context=2, stride=1, seed=None):
    """
    Create a training, dev and test set out of the images in a specified directory.

    Args:
        dataset_dir: A directory containing subdirectories images/ and masks/ with input images
            and masks, respectively.
        image_size: [image height, image width]
        batch_size: Number of images in one batch
        context: How many pixels to add around each strip (the strip then has height 1+context*2)
        stride: Number of pixels between consecutive patches; 1 means no overlap and no skip
        seed: The random seed to use (for reproducibility)

    Returns:
        A dictionary containing the datasets and their names.
    """

    # Names of the masks (end with .png) and their images (.jpg)
    names = [x[:-(len(".png"))] for x in get_image_names(dataset_dir)]

    if not names:
        raise ValueError("No image names found in {}".format(dataset_dir))

    random.seed(seed)
    random.shuffle(names)
    split = [0.8, 0.1, 0.1]
    fr = 0
    datasets = []
    dataset_image_names = []
    for i, (part, name) in enumerate(zip(split, names)):
        if i == len(split) - 1:
            to = len(names)
        else:
            to = fr + int(len(names) * part)
        dataset_image_names.append(names[fr:to])
        datasets.append(
            make_dataset(dataset_dir,
                         names[fr:to],
                         image_size,
                         batch_size=batch_size,
                         shuffle=(True if i == 0 else False),  # only shuffle the training set
                         context=context,
                         stride=stride,
                         repeat=(False if i == 2 else True))  # do not repeat the test set
        )
        fr = to

    return {
        "datasets": datasets,
        "image_names": dataset_image_names,
    }


def make_dataset(dataset_dir, names, image_size, batch_size, shuffle, context, stride, repeat):
    """
    Creates one dataset out of the files in `dataset_dir` with names `names`.

    Args:
        dataset_dir: A directory containing subdirectories images/ and masks/ with input images
            and masks, respectively.
        names: Names of the images out of which to build the dataset
        image_size: [image height, image width]
        batch_size: Number of images in one batch
        shuffle: Whether to shuffle this dataset.
        context: How many pixels to add around each strip (the strip then has height 1+context*2)
        stride: Number of pixels between consecutive patches; 1 means no overlap and no skip
        repeat: Whether to repeat this dataset (we don't want to repeat it if we're predicting).

    Returns:
        The constructed tf.data.Dataset object
    """

    # We assume that if the mask exists, the image exists as well.
    image_filenames = tf.constant([os.path.join(dataset_dir, "images", s + ".jpg") for s in names])
    mask_filenames = tf.constant([os.path.join(dataset_dir, "masks", s + ".png") for s in names])

    # Each element of the datset is a tuple of an image and mask filename.
    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))
    if shuffle:
        # Shuffling strings should be cheap enough
        dataset = dataset.shuffle(10000)

    dataset = (dataset
               .map(_load_image_and_mask(image_size))
               .flat_map(_extract_patches(image_size, context, stride))
               )
    if shuffle:
        dataset = dataset.shuffle(1000)

    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    return dataset


def _load_image_and_mask(image_size):
    """
    Create a function for loading images of a specified size.
    """

    def _load_image_and_mask_wrapped(image_filename, mask_filename):
        image_decoded = tf.image.decode_jpeg(tf.read_file(image_filename), channels=3)
        # Dividing image values by 255 reduces the performance for some reason.
        image_resized = tf.image.rgb_to_grayscale(tf.image.resize_images(image_decoded, image_size))

        mask_decoded = tf.image.decode_jpeg(tf.read_file(mask_filename), channels=3)
        mask_resized = tf.image.resize_images(mask_decoded, image_size)

        return image_resized, mask_resized

    return _load_image_and_mask_wrapped


def _extract_patches(image_size, context, stride):
    """
    Return a function for extracting patches out of an image with a specified context and stride.
    """

    def _extract_patches_wrapped(image, mask):
        """
        Given an image and a mask, extract horizontal patches ("strips") which will be used
        for network training.
        """
        patch_height = 1 + context * 2
        image_patches = tf.extract_image_patches([image],
                                                 ksizes=[1, patch_height, image_size[1], 1],
                                                 strides=[1, patch_height + stride - 1, 1, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding="VALID")
        reshaped_image_patches = tf.reshape(image_patches, [-1, patch_height, image_size[1], 1])

        mask_patches = tf.extract_image_patches([mask],
                                                ksizes=[1, patch_height, image_size[1], 1],
                                                strides=[1, patch_height + stride - 1, 1, 1],
                                                rates=[1, 1, 1, 1],
                                                padding="VALID")
        reshaped_mask_patches = tf.reshape(mask_patches, [-1, patch_height, image_size[1], 3])
        # For some reason, channels seem to get reversed by the patch extraction
        reshaped_mask_patches = tf.reverse(reshaped_mask_patches, axis=[-1])

        # In masks, we only care about the center row, which is what the network is supposed to predict.
        cropped_mask_patches = tf.image.crop_to_bounding_box(
            reshaped_mask_patches, context, context, 1, image_size[1] - context * 2
        )

        # The input masks are coded as follows: outside = #000, black = #F00, white = #FFF
        # This makes the labels [outside, black, white]
        labels = tf.minimum(
            2,
            tf.reduce_sum(tf.cast(tf.greater(cropped_mask_patches, 0), tf.int32), axis=-1)
        )
        # This swaps labels 0 and 1 so we get [black, outside, white]. Simplifies visualization.
        labels = (4 - labels * 2) % 3
        labels = tf.squeeze(labels, axis=[1])  # Get rid of the height dimension, which is of size 1 anyway

        return tf.data.Dataset.from_tensor_slices((reshaped_image_patches, labels))

    return _extract_patches_wrapped


def get_image_names(dataset_dir):
    """
    List mask images in a local on Google Cloud Storage dataset directory.

    Args:
        dataset_dir: Images in [dataset_dir]/masks will be listed

    Returns:
        A list of images in [dataset_dir]/masks
    """

    if dataset_dir.startswith("gs://"):
        # Google Cloud Storage
        dataset_dir = dataset_dir[len("gs://"):]
        bucket_name = dataset_dir[:dataset_dir.index("/")]
        prefix = os.path.join(dataset_dir[dataset_dir.index("/") + 1:], "masks/")

        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)

        return [blob.name[len(prefix):] for blob in blobs]
    else:
        # Local files
        return os.listdir(os.path.join(dataset_dir, "masks"))
