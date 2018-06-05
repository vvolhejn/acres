import argparse
import json
import os
import random

import tqdm
import numpy as np
import cv2

IMAGE_SIZE = (600, 800)  # (height, width)
dataset_parts = [0.8, 0.1, 0.1]


def get_good_rows(mask):
    good_rows = np.where(np.max(mask, axis=1))
    row_from = np.min(good_rows)
    row_to = np.max(good_rows)
    return (row_from, row_to)


def concat_images(img_names, mask_dir, image_dir):
    cropped_masks = []
    cropped_images = []
    for img_name in img_names:
        img_name_no_ext = "".join(img_name.split(".")[:-1])
        mask = cv2.imread(os.path.join(mask_dir, "{}.png".format(img_name_no_ext)), 0)
        img = cv2.imread(os.path.join(image_dir, "{}.jpg".format(img_name_no_ext)), 0)
        row_from, row_to = get_good_rows(mask)
        cropped_masks.append(mask[row_from:row_to + 1, :])
        cropped_images.append(img[row_from:row_to + 1, :])

    res_mask = np.concatenate(cropped_masks, axis=0)
    res_img = np.concatenate(cropped_images, axis=0)
    # cv2.imshow('image', res_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return res_mask, res_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-dir", type=str, help="Path to masks directory")
    parser.add_argument("--image-dir", type=str, help="Path to images directory")
    parser.add_argument("--output-name", type=str, help="Where to store output")
    args = parser.parse_args()

    random.seed(123)
    img_names = os.listdir(args.mask_dir)  # assumes no subdirectories
    random.shuffle(img_names)
    fr = 0
    for i, part in enumerate(dataset_parts):
        start = int(fr * len(img_names))
        end = int((fr + part) * len(img_names))
        res_mask, res_img = concat_images(img_names[start:end], args.mask_dir, args.image_dir)

        cv2.imwrite("{}_{}_mask.png".format(args.output_name, i + 1), res_mask)
        cv2.imwrite("{}_{}.jpg".format(args.output_name, i + 1), res_img)
        fr += part

if __name__ == '__main__':
    main()
