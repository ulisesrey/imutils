import torch

# Check if CUDA is available and display the GPU information
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    for i in range(gpu_count):
        print(f"CUDA is available. Using GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import pandas as pd
from imutils.scopereader import MicroscopeDataReader
import dask.array as da
import sys
import os
import argparse
import logging
import tifffile as tiff

# Set up the SAM2 model
checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def segment_object(pil_image, points):
    # Convert PIL Image to numpy array
    image_np = np.array(pil_image)

    # Set the image for the predictor
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(image_np)

        # Prepare the input prompts (points and labels)
        input_points = np.array(points)  # points should be a list of (x, y) tuples
        input_labels = np.ones(len(points))  # 1 for each point, indicating foreground

        # Predict the mask using multiple points
        masks, _, _ = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=False
        )

    return masks[0]  # Return the first (and only) mask

def extract_coordinates(data, bodyparts):
    result = {}
    for bodypart in bodyparts:
        if bodypart in data.columns.get_level_values('bodyparts'):
            x_values = data[bodypart]['x']
            y_values = data[bodypart]['y']
            result[bodypart] = list(zip(x_values, y_values))
    return result

def main(args):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process input files and generate output.")
    parser.add_argument("input_file_path", type=str, help="Path to the input file")
    parser.add_argument("output_file_path", type=str, help="Path to the output file")
    parser.add_argument("DLC_csv_file_path", type=str, help="Path to the DLC CSV file")
    parser.add_argument("column_names", type=str, nargs='+', help="List of column names")

    # Parse the arguments
    args = parser.parse_args(args)

    # Access the parsed arguments
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    DLC_csv_file_path = args.DLC_csv_file_path
    column_names = args.column_names

    # TODO: Add your main logic here
    print(f"Input file: {input_file_path}")
    print(f"Output file: {output_file_path}")
    print(f"DLC CSV file: {DLC_csv_file_path}")
    print(f"Column names: {column_names}")

    DLC_data = pd.read_csv(DLC_csv_file_path)

    extracted_coordinates = extract_coordinates(DLC_data, column_names)

    try:
        if os.path.isdir(input_file_path):
            reader_obj = MicroscopeDataReader(input_file_path)
        elif os.path.isfile(input_file_path):
            reader_obj = MicroscopeDataReader(input_file_path, as_raw_tiff=True, raw_tiff_num_slices=1)
        else:
            raise ValueError("Invalid input file path. Please provide a valid directory or file path.")

        tif = da.squeeze(reader_obj.dask_array)

        with tiff.TiffWriter(output_file_path, bigtiff=True) as tif_writer:
            total_frames = len(tif)
            for i, img in enumerate(tif):
                logging.info(f"Processing image {i+1}/{total_frames}")
                img = np.array(img)

                mask = segment_object(img, [extracted_coordinates[column][i] for column in column_names])

                tif_writer.write(mask, contiguous=True)
                
                logging.info(f"Successfully processed and wrote image {i+1}/{total_frames}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == '__main__':
    main(sys.argv[1:])  # exclude the script name from the args when called from sh
