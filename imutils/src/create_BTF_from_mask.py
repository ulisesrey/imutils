import os
import numpy as np
import argparse
import pandas as pd
from PIL import Image
import tifffile as tf

def crop_BTF(tiff_stack, x_roi_data, y_roi_data, crop_size):
    # Read the big TIFF stack
    img_stack = tf.imread(tiff_stack)
    depth, height, width = img_stack.shape

    # Create an empty list to store cropped images
    new_roi_array = []

    for frame_number in range(depth):
        print(f"Processing frame {frame_number} of {depth}")

        # Ensure the frame number is within the range of x_roi_data and y_roi_data
        if frame_number < len(x_roi_data):
            roi_x, roi_y = x_roi_data[frame_number], y_roi_data[frame_number]
        else:
            print("Frame number is out of range of ROI data. Skipping frame.")
            continue

        roi_width, roi_height = crop_size

        # Ensure ROI coordinates are within frame boundaries, else skip
        if (0 <= roi_x < width and 0 <= roi_y < height and
                roi_x + roi_width <= width and roi_y + roi_height <= height):
            # Extract the ROI from the original frame
            frame_roi = img_stack[frame_number, roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            new_roi_array.append(frame_roi)
        else:
            print(f"Frame {frame_number}: ROI is out of bounds. Skipping frame...")

    return new_roi_array

def export_BTF(cropped_stack, output, crop_size):
    roi_width, roi_height = crop_size

    # Construct the output file path
    base, ext = os.path.splitext(output)
    output_file = f"{base}_cropped{ext}"

    # Convert list to numpy array
    cropped_stack_array = np.array(cropped_stack)

    # Save as big TIFF
    tf.imwrite(output_file, cropped_stack_array, bigtiff=True)

    print(f"Exported cropped stack to {output_file}")

def read_from_excel(excel_file):
    # Specify the engine 'openpyxl' to read xlsx files
    df = pd.read_excel(excel_file, engine='openpyxl')
    # Convert the 'x_roi' and 'y_roi' columns to lists and return them
    return df['x_roi'].tolist(), df['y_roi'].tolist()

def main(arg_list=None):
    parser = argparse.ArgumentParser(description="Process and export cropped BigTIFF stacks")
    parser.add_argument("--stack_path", required=True)
    parser.add_argument("--excel_file", required=True, help="Input Excel file with ROI data")
    parser.add_argument("--crop_size", required=True, help="Crop size in format 'width,height'")
    args = parser.parse_args(arg_list)

    stack_path = args.stack_path
    excel_file = args.excel_file
    crop_size = tuple(map(int, args.crop_size.split(',')))

    print("Reading ROI data from Excel file...")
    x_roi_data, y_roi_data = read_from_excel(excel_file)

    print("Processing BigTIFF stack...")
    cropped_stack = crop_BTF(stack_path, x_roi_data, y_roi_data, crop_size)

    print("Exporting BigTIFF stack...")
    export_BTF(cropped_stack, stack_path, crop_size)

if __name__ == "__main__":
    import sys
    print("Shell commands passed:", sys.argv)
    main(sys.argv[1:])  # exclude the script name from the args when called from shell
