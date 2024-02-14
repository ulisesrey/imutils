import tifffile as tiff
import cv2
import numpy as np
import matplotlib.pyplot as plt

def stack_make_binary(stack_input_filepath: str, stack_output_filepath: str, lower_threshold: float,
                      higher_threshold: float):
    """
    write a binary stack based on lower and higher threshold
    Parameters:
    -------------
    stack_input_filepath, str
    stack_output_filepath, str
    lower_threshold, float
    higher_threshold, float
    Returns:
    -------------
    None
    """
    with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer, tiff.TiffFile(stack_input_filepath) as tif:
        for i, page in enumerate(tif.pages):
            img = page.asarray()
            # apply threshold
            ret, new_img = cv2.threshold(img, lower_threshold, higher_threshold, cv2.THRESH_BINARY)
            #convert matrix to np.uint
            new_img = new_img * 255
            new_img = new_img.astype(np.uint8)
            tif_writer.write(new_img, contiguous=True)



if __name__ == "__main__":
    stack_input_filepath = "/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221210/data/ZIM2165_Gcamp7b_worm6/2022-12-10_19-00_ZIM2165_worm6_Ch0-BH/2022-12-10_19-00_ZIM2165_worm6_Ch0-BHbigtiff_AVG_background_subtracted_normalised_unet_segmented_weights_5358068_1.btf"
    stack_output_filepath = "/Users/ulises.rey/local_data/binarize_test/binary.btf"
    lower_threshold = 0.05
    higher_threshold = 1

    stack_make_binary(stack_input_filepath, stack_output_filepath, lower_threshold, higher_threshold)