import os
import argparse

import tqdm
import cv2
import numpy as np

DEBUG = False


def show(img):
    cv2.imshow("image", img)
    cv2.waitKey(0)


def strip_extension(s):
    return os.path.splitext(s)[0]


def main(images_dir, masks_dir, output_dir, mask_weight=1):
    image_names = set(strip_extension(f) for f in os.listdir(images_dir))
    mask_names = set(strip_extension(f) for f in os.listdir(masks_dir))
    common_image_names = image_names & mask_names
    if not common_image_names:
        print("No images in common!")
        exit(1)

    for name in tqdm.tqdm(common_image_names):
        img = cv2.imread(os.path.join(images_dir, "{}.jpg".format(name)))
        mask = cv2.imread(os.path.join(masks_dir, "{}.png".format(name)))

        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        mask = mask / mask.max()
        img = img / img.max()
        mask = (mask[:, :, 0] * -1. + mask[:, :, 1] * 0. + mask[:, :, 2] * 1.) + 1
        masked = img * (1 - mask_weight) + (2 - mask[:, :, np.newaxis]) * 0.5 * mask_weight  # * mask_weight
        # masked = np.clip((masked - 1) * 2, 0, 1)

        if DEBUG:
            while True:
                show(img)
                show(masked)
                show(mask)

        cv2.imwrite(os.path.join(output_dir, "{}.jpg".format(name)), masked * 255)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_dir", help="Directory containing image files")
    parser.add_argument("masks_dir", help="Directory containing mask files to overlay the images with")
    parser.add_argument("output_dir", help="Directory in which to place the output files")
    parser.add_argument("--mask-weight", type=float, help="What importance is given to the mask")

    args = parser.parse_args()
    main(args.images_dir, args.masks_dir, args.output_dir, args.mask_weight)
