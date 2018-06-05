import argparse
import json
import os

import tqdm
import numpy as np
import cv2

IMAGE_SIZE = (600, 800)  # (height, width)


def show_masked(path, mask):
    img = cv2.imread(path, 0)
    img = img * (mask != 0)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def make_mask(xs, ys):
    img = np.zeros(IMAGE_SIZE, np.uint8)  # .transpose()
    pts = np.array([xs, ys], np.int32).transpose()
    cv2.fillPoly(img, [pts], 255)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, help="Path to labels file")
    parser.add_argument("--input-dir", type=str, help="Path to images")
    parser.add_argument("--output-dir", type=str, help="Where to store output")
    args = parser.parse_args()

    with open(args.labels, "r") as f:
        data = json.load(f)

    for img_name, img_data in tqdm.tqdm(data.items()):
        if not img_data["regions"]:
            continue
        polygon = img_data["regions"]["0"]["shape_attributes"]
        mask = make_mask(polygon["all_points_x"], polygon["all_points_y"])
        cv2.imwrite(os.path.join(args.output_dir, img_name.split(".")[0] + ".png"), mask)


if __name__ == '__main__':
    main()
