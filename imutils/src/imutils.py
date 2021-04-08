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
    Convert a stack into a folder with all the images, saving each image with its original name
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
        for idx,page in enumerate(tif.pages):
            img=page.asarray()
            filename=files[idx]
            tiff.imsave(os.path.join(output_path, filename), img)         

def max_projection_3d(input_filepath, output_filepath, fold_increase=3, nplanes=20):

    """
    Create a visualization image of a volume, with the 3 max projections possible.

    Parameters:
    ------------
    input_filepath: str,

    output_filepath: str,

    fold_increase: int, (it could be float, but code should be tested)
        Expands the z dimension so that the image has crrect dimensions. Depends on the ratio between xy pixel size and z-step size.
    nplanes: int,

    """
    print('creating writer object..')
    with tiff.TiffWriter(output_filepath, bigtiff=True) as output_tif:
        print('creating reader object..')
        with tiff.TiffFile(input_filepath) as tif:
            print('looping through reader object..')
            for idx, page in enumerate(tif.pages):#enumerate(tif.series):#enumerate(tif.series[0].pages):
                if idx%3300==0: print(idx)
                img=page.asarray()
                #if it is the first plane of the volume create an empty img_stack shape=(w, h, z)
                if idx%nplanes==0:
                    #print('index is ',idx)
                    img_stack=np.full(shape=(img.shape[0], img.shape[1], nplanes), fill_value=np.nan, dtype=np.uint16)

                #fill the idx plane with the img
                img_stack[:,:,idx%nplanes]=img

                #if it is the last plane on the volume do max projection on the three axis and concatenate them
                if idx%nplanes==nplanes-1:
                    max0 = np.max(img_stack, axis=0)
                    max1 = np.max(img_stack, axis=1)
                    max2 = np.max(img_stack, axis=2)



                    # extends the YZ and XZ max projection for better visualization based on input fold_increase variable
                    max0= np.repeat(max0, fold_increase, axis=1)
                    # rotates IM_MAX_2
                    max0=np.transpose(max0)
                    max1 = np.repeat(max1, fold_increase, axis=1)

                    #defines corner array
                    corner_matrix=np.full((fold_increase*img_stack.shape[2],fold_increase*img_stack.shape[2]),100, dtype='uint16')

    #                 # concatenate the different max projections into one image
    #                 plt.imshow(rep_max1)
    #                 plt.title('repmax1')
    #                 plt.show()
                    vert_conc_1 = cv2.hconcat([max2,max1])
    #                 plt.imshow(vert_conc_1)
    #                 plt.title('repmax1,max2')
    #                 plt.show()
                    #np.concatenate((max0, rep_max1), axis=0)
                    vert_conc2 = cv2.hconcat([max0, corner_matrix])#np.concatenate((rep_rot_max0, corner_matrix), axis=0)
                    final_img = cv2.vconcat([vert_conc_1,vert_conc2])#np.concatenate((vert_conc_1, vert_conc2), axis=1)
                    #print(final_image.shape)
    #                 plt.figure(dpi=300)
    #                 plt.imshow(final_image)
    #                 plt.show()
                    #break
                    #save the 3 max projection image
                    output_tif.save(final_img, photometric='minisblack')





