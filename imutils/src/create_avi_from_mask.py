# This script processes a video file by cropping it based on regions of interest (ROIs) defined in an Excel file ('roi_data').

import os
import numpy as np
import cv2
import argparse
import pandas as pd
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows

def crop_avi_as_well(video, x_roi_data, y_roi_data, fps, crop_size):
    # Open the video file
    cap = cv2.VideoCapture(video)

    # Create an empty NumPy array to store grayscale values
    new_roi_array = []

    while True:
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))  # Get the current frame number
        ret, frame = cap.read()  # Read the next frame

        if not ret:
            break  # Break the loop when there are no more frames

        # Debugging print statements
        print("Frame number (from video):", frame_number)
        print("Type of frame_number:", type(frame_number))
        print("Length of x_roi_data:", len(x_roi_data))

        # Ensure the frame number is within the range of x_roi_data and y_roi_data
        if frame_number < len(x_roi_data):
            roi_x = x_roi_data[frame_number]
            roi_y = y_roi_data[frame_number]
        else:
            print("Frame number is out of range of ROI data. Skipping frame.")
            continue

        roi_width, roi_height = crop_size

        # Ensure ROI coordinates are within frame boundaries, else skip
        if (
                roi_x >= 0 and
                roi_y >= 0 and
                roi_x + roi_width <= frame.shape[1] and
                roi_y + roi_height <= frame.shape[0]
        ):
            # Extract the grayscale ROI from the original frame
            #frame_roi = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
            frame_roi = cv2.cvtColor(frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width], cv2.COLOR_BGR2GRAY)
            # Append the grayscale values of the ROI to the list
            new_roi_array.append(frame_roi)
        else:
            # Create an empty frame (all black) when the ROI is out of bounds
            empty_frame = np.zeros((roi_height, roi_width), dtype=np.uint8)
            # Append the empty frame to the list
            new_roi_array.append(empty_frame)
            # Print a message indicating that an empty frame is added
            print(f"Frame {frame_number}: ROI is out of bounds. Adding an empty frame...")

    # Release the video capture object and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    return new_roi_array

def export_video(cropped_video_stack, output, frame_rate, crop_size):
    roi_width, roi_height = crop_size

    #generating new video name "video_cropped.avi"

    # Splitting the path and getting the file name and extension
    video_dir, video_file = os.path.split(output)
    video_name, video_extension = os.path.splitext(video_file)

    # Creating the new file name with "_cropped" suffix
    new_video_name = video_name + "_cropped" + video_extension

    # Creating the new video output path
    video_output = os.path.join(video_dir, new_video_name)

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
    df = pd.read_excel(excel_file)
    return df['x_roi'].tolist(), df['y_roi'].tolist()

def main(arg_list=None):
    parser = argparse.ArgumentParser(description="Process and export cropped video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", required=True)
    parser.add_argument("--excel_file", required=True, help="Input Excel file with ROI data")
    parser.add_argument("--crop_size", required=True, help="Crop size in format 'width,height'")
    args = parser.parse_args()

    video_path = args.video
    fps = float(args.fps)
    excel_file = args.excel_file
    crop_size = tuple(map(int, args.crop_size.split(',')))

    print("Reading ROI data from Excel file...")
    x_roi_data, y_roi_data = read_from_excel(excel_file)

    print("Processing video...")
    cropped_video_stack = crop_avi_as_well(video_path, x_roi_data, y_roi_data, fps, crop_size)

    print("Exporting video...")
    export_video(cropped_video_stack, video_path, fps, crop_size)


if __name__ == "__main__":

    print("Shell commands passed:", sys.argv)
    main(sys.argv[1:])  # exclude the script name from the args when called from shell
