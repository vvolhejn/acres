import os
import random

import tensorflow as tf


def make_dataset(dataset_dir, names):
    """
    Creates one dataset out of the files in `dataset_dir` with names `names`.
    """
    size = [200, 200]

    def _parse_function(image_filename, mask_filename):
        image_decoded = tf.image.decode_jpeg(tf.read_file(image_filename), channels=3)
        image_resized = tf.image.rgb_to_grayscale(tf.image.resize_images(image_decoded, size))

        mask_decoded = tf.image.decode_jpeg(tf.read_file(mask_filename), channels=1)
        mask_resized = tf.image.resize_images(mask_decoded, size)
        return image_resized, mask_resized

    image_filenames = tf.constant([os.path.join(dataset_dir, "images", s + ".jpg") for s in names])
    mask_filenames = tf.constant([os.path.join(dataset_dir, "masks", s + ".png") for s in names])

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))
    dataset = dataset.map(_parse_function)
    return dataset.shuffle(1000).batch(10)


def make_datasets(dataset_dir):
    """
    Takes all masks from `dataset_dir`/masks and their image counterparts
    in `dataset_dir`/images and splits them into a train, dev and test set.
    Returns a tuple: (training set, dev set, test set)
    """

    # Names of the masks (end with .png) and their images (.jpg)
    names = [x[:-(len(".png"))] for x in os.listdir(os.path.join(dataset_dir, "masks"))]
    random.shuffle(names)

    split = [0.8, 0.1, 0.1]

    fr = 0
    datasets = []
    for i, (part, name) in enumerate(zip(split, names)):
        if i == len(split) - 1:
            to = len(names)
        else:
            to = fr + int(len(names) * part)

        datasets.append(make_dataset(dataset_dir, names[fr:to]))
        fr = to

    return datasets


# print(make_datasets("data/"))
