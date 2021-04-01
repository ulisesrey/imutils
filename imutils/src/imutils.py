import cv2
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv

import os
from natsort import natsorted
import re


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
                tiff.imwrite(os.path.join(output_folder,'img'+str(i)+'.tif'),img)
                
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

def images2stack_RAM(path, output_filename):
    """
    Convert the images in one folder into a stack, keeping their filenames in the metadata.
    It is not an object to it needs to allocate all the memory for the stack
    Parameters:
    -------------
    path: str, path to the input_folder
    output_filename: str, name of the output stack
    """
    files = os.listdir(path)#tifffile.natural_sorted(output_path)
    image = tifffile.imread(os.path.join(path,files))

def images2stack(path, output_filename):
    """
    Convert the images in one folder into a stack, keeping their filenames
    Parameters:
    -------------
    path: str, path to the input_folder
    output_filename: str, name of the output stack
    """
    files = natsorted(os.listdir(path))
    metadata = {'Info': '\n'.join(files)}

    with tiff.TiffWriter(output_filename, imagej=True) as tif:
        for filename in files:
            image = tiff.imread(os.path.join(path,filename))
            tif.write(image, contiguous=True, photometric='minisblack', metadata=metadata)
            metadata = None
            
def stack2images(input_filename, output_path):
    """
    Convert a stack into a folder with all the images
    Parameters:
    -------------
    input_filename:str, name of the input stack
    output_path: str, path to the directory where it will be saved


    If it does not work check here: https://forum.image.sc/t/keep-image-description-metadata-in-a-stack-after-modifying-it/50625/4
    """
    try: os.mkdir(output_path) # creates the subdirectory where it should be stored
    except: print('Output Directory already exists, might overwrite')
    with tiff.TiffFile(input_filename) as tif:
        files = tif.imagej_metadata['Info'].split('\n')
        image = tif.asarray()

    for i, fname in enumerate(files):
        tiff.imsave(os.path.join(output_path, fname), image[i])






