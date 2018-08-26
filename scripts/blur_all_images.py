"""
Take a directory of images and produce versions of each image with varying levels of blur.
Uses ImageMagick: http://www.imagemagick.org/Usage/blur/
"""
import argparse
import os
import subprocess
from tqdm import tqdm


def blur_images():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    for in_name in tqdm(os.listdir(args.input_dir)):
        in_path = os.path.join(args.input_dir, in_name)
        # The levels of blur are hard-coded in the range.
        for blur_10 in range(5, 35, 2):
            blur = blur_10 / 10
            out_path = os.path.join(args.output_dir, "{:0>2}_{}".format(blur_10, in_name))
            subprocess.call(["convert", in_path, "-blur", "x{}".format(blur), out_path])


if __name__ == '__main__':
    blur_images()
