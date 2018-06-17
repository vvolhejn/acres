"""
The Muenster DB image masks were 640x480 and we want 800x600
"""
import argparse
import os

import cv2


def show_masked(path):
    print(path)
    img = cv2.imread(path)
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks-dir", type=str, help="Path to masks")
    parser.add_argument("--output-dir", type=str, help="Where to store output")
    args = parser.parse_args()

    for name in os.listdir(args.masks_dir):
        img = cv2.imread(os.path.join(args.masks_dir, name))
        resized_image = cv2.resize(img, (800, 600))
        cv2.imwrite(os.path.join(args.output_dir, name.replace("640x480", "800x600")), resized_image)


if __name__ == '__main__':
    main()
