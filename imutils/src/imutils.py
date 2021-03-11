import cv2
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import os
from natsort import natsorted
import re
import argparse

def extract_frames(input_image, output_folder, frames_list):
    """
    Extract the frames from an stack (image) given a list, and save them in the output_folder

    Parameters:
    -----------
    input_image: str,
        Path to the input_image (stack)
    output_folder: str
        Path to the folder where the images will be saved
    frames_list: list?
        list of integers, frames that will be extracted
    """
    
    with tiff.TiffFile(input_image, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            #if the image is not on the frames_list then skip
            if i in frames_list:
                img=page.asarray()
                #print(os.path.join(output_folder,'img'+str(i)+'.'+str(file_format)))
                tiff.imwrite(os.path.join(output_folder,'img'+str(i)+'.tiff'),img)