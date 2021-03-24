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
    if not os.path.exists(output_folder):
        
        print('making ', output_folder, ' directory')
        os.makedirs(output_folder)
    else: print(output_folder, 'already exists')

    
    with tiff.TiffFile(input_image, multifile=False) as tif:
        for i, page in enumerate(tif.pages):
            #if the image is not on the frames_list then skip
            if i in frames_list:
                img=page.asarray()
                #print(os.path.join(output_folder,'img'+str(i)+'.'+str(file_format)))
                tiff.imwrite(os.path.join(output_folder,'img'+str(i)+'.tiff'),img)
                
def add_zeros_to_filename(path, len_max_number=6):
    """
    Change the filename of images from img235.png to img00235.png depending on len_max_number
    It has a sister function: add_zeros_to_csv
    Parameters:
    -----------
    path: str,
        Path to the directory with the images
    len_max_number: int
        number of digits the number should have, default is 6
    """
    #creates numberic regular expression
    regex_num=re.compile(r'\d+')

    files=os.listdir(path)
    #this could be improved with a regular expression to catch the numbers.
    for filename in files:
        if 'img' not in filename: continue
        #print(filename)
        new_filename=filename
        
        number=regex_num.search(filename).group(0)

        #get the file exntesion without the do (e.g. 'png') 
        file_extension=re.split('\.',filename)[-1]
        
        #while the number is smaller than the len_max_number
        while len(number) < len_max_number:
            #print(re.split('img', filename)[1])
            number='0'+number
        #new filename= img string+number+ dot + extension
        new_filename='img'+number+'.'+file_extension
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

def images2stack(path, output_filename):
    """
    Convert the images in one folder into a stack, keeping their filenames
    Parameters:
    -------------
    path: str, path to the input_folder
    output_filename: str, name of the output stack
    """
    with tiff.TiffWriter(output_filename, bigtiff=True) as tif_writer:
        for idx, filename in enumerate(natsorted(os.listdir(path))):
            img=tiff.imread(os.path.join(path,filename),name=filename)
            tif_writer.save(img, photometric='minisblack', description=filename, metadata=None)
            
def stack2images(input_filename, output_path):
    """
    Convert a stack into a folder with all the images
    Parameters:
    -------------
    input_filename:str, name of the input stack
    output_path: str, path to the directory where it will be saved
    """
    try: os.mkdir(output_path) # creates the subdirectory where it should be stored
    except: print('Output Directory already exists, might overwrite')
    with tiff.TiffFile(input_filename, multifile=True) as tif:
        for page in tif.pages:
            img=page.asarray()
            description=page.description
            tiff.imsave(file=os.path.join(output_path,description), data=img, photometric='minisblack')