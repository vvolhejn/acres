import argparse
import os

import numpy as np
import cv2


def show(img):
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()


def is_vertical(img):
    # Are the _bars_ vertical?
    horiz = np.array([[-1, -1, 4, -1, -1]])
    vert = np.array([[-1], [-1], [4], [-1], [-1]])
    hc = cv2.filter2D(img, -1, horiz)  # Convolve
    vc = cv2.filter2D(img, -1, vert)  # Convolve
    res = np.mean(hc) > np.mean(vc)

    print(res, np.mean(hc) - np.mean(vc))
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, help="Path to images")
    parser.add_argument("--masks-dir", type=str, help="Path to masks")
    parser.add_argument("--output-dir", type=str, help="Where to store output")
    args = parser.parse_args()

    names = [x[:-len(".png")] for x in os.listdir(args.masks_dir)]
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)

    for name in names:
        image = cv2.imread(os.path.join(args.images_dir, name + ".jpg"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(args.masks_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
        _, mask_bw = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # cv2.imshow("image", image)
        # cv2.waitKey(0)
        image_binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 21, 1)
        image_rgb = cv2.cvtColor(image_binary, cv2.COLOR_GRAY2RGB)
        image_rgb[:, :, 0] = 255  # Seems to be blue here.
        image_rgb = image_rgb * np.expand_dims(mask_bw != 0, 2)

        # cv2.imshow("image", image_rgb)
        # cv2.waitKey(0)
        if is_vertical(image_rgb):
            cv2.imwrite(os.path.join(args.output_dir, name + ".png"), image_rgb)


if __name__ == '__main__':
    main()
