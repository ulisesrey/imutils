import os
import tifffile
import numpy as np
from skimage import measure
import cv2
import argparse

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

        roi_x = x_roi_data[frame]
        roi_y = x_roi_data[frame]

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
    video_dir, video_file = os.path.split(video_path)
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

def process_frame(frame, crop_size):

    # Segment the image
    labeled_frame = measure.label(frame)
    regions = measure.regionprops(labeled_frame)

    # Sort regions by area and get the largest
    regions.sort(key=lambda x: x.area, reverse=True)
    largest_region = regions[0]

    # Get the centroid of the largest region
    centroid_y, centroid_x = largest_region.centroid

    # Calculate the top-left corner of the crop area
    start_x = int(max(centroid_x - crop_size // 2, 0))
    start_y = int(max(centroid_y - crop_size // 2, 0))

    # Crop the image
    cropped_frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

    return cropped_frame, start_x, start_y

def crop_tiff(path, output_path, crop_size):

    print(f"Cropping: {path}")

    x_roi_data = []
    y_roi_data = []

    with tifffile.TiffFile(path) as tif:
        total_frames = len(tif.pages)

        start_frame = 0
        end_frame = total_frames

        # Create a new filename with '_cropped' appended before the file extension
        base = os.path.basename(path)
        name, ext = os.path.splitext(base)
        output_filename = f"{name}_cropped{ext}"

        output_file_path = os.path.join(output_path, output_filename)  # Full path for the output file

        # Read and process each frame within the specified range
        cropped_data = []

        for i in range(start_frame, end_frame):
            frame = tif.asarray(key=i)
            cropped_frame, start_x, start_y = process_frame(frame, crop_size)
            cropped_data.append(cropped_frame)

            x_roi_data.append(start_x)
            y_roi_data.append(start_y)

        # Save the cropped frames to a new TIFF file
        tifffile.imsave(output_file_path, np.array(cropped_data), bigtiff=True)


    print(f"Output saved to: {output_file_path}")

    return x_roi_data, y_roi_data

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="crop_worm_from_binary_mask_tiff")
    parser.add_argument("--binary_stack", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop_size", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--fps", required=True)
    args = parser.parse_args()

    path = args.binary_stack
    output = args.output
    crop_size = int(args.crop_size)
    video_path = args.video
    fps = args.fps

    print("Binary Mask Path:", path)
    print("Outputfile:", output)

    output_folder =  output.split('/')

    print("output_folder:",  output_folder[0])

    x_roi_data, y_roi_data = crop_tiff(path, output, crop_size)

    cropped_video_stack = crop_avi_as_well(video_path, x_roi_data, y_roi_data, fps, crop_size)
    export_video(cropped_video_stack, video_path, fps, crop_size)