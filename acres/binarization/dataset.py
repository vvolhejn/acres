import os
import random

import tensorflow as tf


def get_image_names(dataset_dir):
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


def make_dataset(dataset_dir, names, image_size, context=0, stride=1):
    """
    Creates one dataset out of the files in `dataset_dir` with names `names`.
    `image_size` = [image height, image width]
    `context` = how many pixels to add around each strip (the strip then has height 1+context*2)
    """
    size = [200, 200]

    def _parse_function(image_filename, mask_filename):
        image_decoded = tf.image.decode_jpeg(tf.read_file(image_filename), channels=3)
        image_resized = tf.image.rgb_to_grayscale(tf.image.resize_images(image_decoded, image_size))

        mask_decoded = tf.image.decode_jpeg(tf.read_file(mask_filename), channels=1)
        mask_resized = tf.image.resize_images(mask_decoded, image_size)
        return image_resized, mask_resized

    def _slice(image, mask):
        slice_height = 1 + context * 2
        image_patches = tf.extract_image_patches([image],
                                                 ksizes=[1, slice_height, image_size[1], 1],
                                                 strides=[1, slice_height + stride - 1, 1, 1],
                                                 rates=[1, 1, 1, 1],
                                                 padding="VALID")
        reshaped_image_patches = tf.reshape(image_patches, [-1, slice_height, image_size[1]])

        mask_patches = tf.extract_image_patches([mask],
                                                ksizes=[1, slice_height, image_size[1], 1],
                                                strides=[1, slice_height + stride - 1, 1, 1],
                                                rates=[1, 1, 1, 1],
                                                padding="VALID")
        reshaped_mask_patches = tf.reshape(mask_patches, [-1, slice_height, image_size[1]])

        return tf.data.Dataset.from_tensor_slices((reshaped_image_patches, reshaped_mask_patches))

    image_filenames = tf.constant([os.path.join(dataset_dir, "images", s + ".jpg") for s in names])
    mask_filenames = tf.constant([os.path.join(dataset_dir, "masks", s + ".png") for s in names])

    dataset = tf.data.Dataset.from_tensor_slices((image_filenames, mask_filenames))
    dataset = dataset.map(_parse_function).flat_map(_slice)
    return dataset.shuffle(1000).batch(10).repeat()


def make_datasets(dataset_dir, image_size, context=0):
    """
    Takes all masks from `dataset_dir`/masks and their image counterparts
    in `dataset_dir`/images and splits them into a train, dev and test set.
    Returns a tuple: (training set, dev set, test set)
    """

    # Names of the masks (end with .png) and their images (.jpg)
    names = [x[:-(len(".png"))] for x in get_image_names(dataset_dir)]
    random.shuffle(names)

    split = [0.8, 0.1, 0.1]

    fr = 0
    datasets = []
    for i, (part, name) in enumerate(zip(split, names)):
        if i == len(split) - 1:
            to = len(names)
        else:
            to = fr + int(len(names) * part)

        datasets.append(make_dataset(dataset_dir, names[fr:to], image_size, 2))
        fr = to

    return datasets


# x = make_datasets("data/", [200, 200], 2)[0]
# sess = tf.Session()
# next_element = x.make_one_shot_iterator().get_next()
# value = sess.run(next_element)
# print(value[0].shape, value[1].shape)

# import cv2
# cv2.namedWindow("image", cv2.WINDOW_NORMAL)
# cv2.imshow('image', value[0][1] / 255)
# cv2.waitKey(0)
