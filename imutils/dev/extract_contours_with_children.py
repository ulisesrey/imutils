from imutils.src import imfunctions
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import os

from imutils.src.imfunctions import crop_image_from_contour







if __name__ == "__main__":
    # run with array_job_directories_extract_contours.sh
    import argparse
    #for booleans in argparse see: https://from-locals.com/python-argparse-bool/

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-bi', '--bi_path', help='binary input filepath', required=True)
    parser.add_argument('-ri', '--ri_path', help='raw input filepath', required=True)
    parser.add_argument('-o', '--o_path', help='output folder', required=True)
    parser.add_argument('-ct', '--crop_true', action='store_true', help='set crop to True', required=False)
    parser.add_argument('-s', '--subsample', help='subsample', type=int, required=False)

    args = vars(parser.parse_args())
    binary_input_filepath = args['bi_path']
    raw_input_filepath = args['ri_path']
    output_folder = args['o_path']
    crop = args['crop_true']
    subsample = args['subsample']

    stack_extract_and_save_contours_with_children(binary_input_filepath, raw_input_filepath, output_folder, crop,
                                                  subsample=subsample)
