# This script processes a video file by cropping it based on regions of interest (ROIs) defined in an Excel file ('roi_data').
import sys
import os
import numpy as np
import cv2
import argparse
import pandas as pd


def crop_avi_as_well(video, x_roi_data, y_roi_data, fps, crop_size):
    # Open the video file
    cap = cv2.VideoCapture(video)

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Adjust the length of ROI data to match the number of frames if necessary
    min_length = min(len(x_roi_data), len(y_roi_data), total_frames)
    x_roi_data = x_roi_data[:min_length]
    y_roi_data = y_roi_data[:min_length]

    # Pre-allocate list with None to improve memory management
    new_roi_array = [None] * min_length

    frame_number = 0  # Initialize frame number
    while True:
        ret, frame = cap.read()  # Read the next frame

        if not ret or frame_number >= min_length:
            break  # Break the loop when there are no more frames or data points

        roi_x, roi_y = x_roi_data[frame_number], y_roi_data[frame_number]
        roi_width, roi_height = crop_size

        # Check if ROI coordinates are within frame boundaries
        if 0 <= roi_x < frame.shape[1] - roi_width and 0 <= roi_y < frame.shape[0] - roi_height:
            # Extract and convert the ROI to grayscale
            frame_roi = cv2.cvtColor(frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], cv2.COLOR_BGR2GRAY)
            new_roi_array[frame_number] = frame_roi
        else:
            # Create an empty frame (all black) and append to the list if the ROI is out of bounds
            new_roi_array[frame_number] = np.zeros((roi_height, roi_width), dtype=np.uint8)

        frame_number += 1  # Increment the frame number

    # Release the video capture object
    cap.release()

    # Remove any remaining None values in case the video was shorter than expected
    new_roi_array = [frame if frame is not None else np.zeros((roi_height, roi_width), dtype=np.uint8) for frame in
                     new_roi_array]

    return new_roi_array


def export_video(cropped_video_stack, output, frame_rate, crop_size):

    roi_width, roi_height = crop_size

    #generating new video name "video_cropped.avi"

    # Splitting the path and getting the file name and extension
    video_dir, video_file = os.path.split(output)
    video_name, video_extension = os.path.splitext(video_file)

    # Creating the new file name with "_cropped" suffix
    new_video_name = video_name + "_cropped" + video_extension

    print("Cropped video name:", new_video_name)

    # Creating the new video output path
    video_output = os.path.join(video_dir, new_video_name)

    print("Videopath:", video_output)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Use the XVID codec for AVI
    out = cv2.VideoWriter(video_output, fourcc, frame_rate, (roi_width, roi_height))

    for frame in cropped_video_stack:
        # Apply histogram equalization
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        # Write the frame to the video file
        out.write(frame)

    # Release the VideoWriter object
    out.release()

    print("Crop Size:", crop_size)


def read_from_excel(excel_file):
    try:
        # Attempt to read the Excel file
        df = pd.read_excel(excel_file, engine='openpyxl')

        # Check if the necessary columns are present
        if 'x_roi' not in df.columns or 'y_roi' not in df.columns:
            # If either column is missing, raise a ValueError
            raise ValueError("Excel file is missing required 'x_roi' or 'y_roi' columns.")

        # Convert the 'x_roi' and 'y_roi' columns to lists and return them
        return df['x_roi'].tolist(), df['y_roi'].tolist()
    except FileNotFoundError:
        # Handle the case where the Excel file doesn't exist
        print(f"Error: The file {excel_file} does not exist.")
        return [], []  # Return empty lists as a fail-safe
    except ValueError as ve:
        # Handle missing column error
        print(f"Error: {ve}")
        return [], []  # Return empty lists as a fail-safe
    except Exception as e:
        # Handle any other exceptions that may occur
        print(f"An unexpected error occurred: {e}")
        return [], []  # Return empty lists as a fail-safe


def main(arg_list=None):
    parser = argparse.ArgumentParser(description="Process and export cropped video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", required=True)
    parser.add_argument("--excel_file", required=True, help="Input Excel file with ROI data")
    parser.add_argument("--crop_size", required=True, help="Crop size in format 'width,height'")
    args = parser.parse_args(arg_list)

    video_path = args.video
    fps = float(args.fps)
    excel_file = args.excel_file
    crop_size = tuple(map(int, args.crop_size.split(',')))

    print("Reading ROI data from Excel file...")
    x_roi_data, y_roi_data = read_from_excel(excel_file)

    print("Processing video...")
    cropped_video_stack = crop_avi_as_well(video_path, x_roi_data, y_roi_data, fps, crop_size)

    print("Number of frames in cropped video stack:", len(cropped_video_stack))

    print("Exporting video...")
    export_video(cropped_video_stack, video_path, fps, crop_size)


if __name__ == "__main__":

    print("Shell commands passed:", sys.argv)
    main(sys.argv[1:])  # exclude the script name from the args when called from shell
