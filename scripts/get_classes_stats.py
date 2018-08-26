import argparse
import os

import cv2
import numpy as np
import tqdm


def main(masks_dir):
    class_totals = [0, 0, 0]
    total_pixels = 0
    for name in tqdm.tqdm(os.listdir(args.masks_dir)):
        img = cv2.imread(os.path.join(args.masks_dir, name))

        img = np.sum(img > 0, axis=2)
        # Relabel classes to [black, outside, white]
        img[img == 1] = 2
        img[img == 0] = 1
        img[img == 3] = 0

        total_pixels += img.size
        for i in range(3):
            class_totals[i] += np.sum(img == i)

    for t, name in zip(class_totals, ["Black", "Outside", "White"]):
        print("{: <9}{:.1f}%".format(name, t / total_pixels * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Given a directory of input mask images, "
                                                  "compute some statistics about the dataset"))
    parser.add_argument("masks_dir", type=str, help="Path to masks")
    args = parser.parse_args()
    main(args.masks_dir)
