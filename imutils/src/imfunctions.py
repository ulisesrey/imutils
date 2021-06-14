import cv2
import tifffile as tiff

import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import re

import pandas as pd
import csv

def tiff2avi(tiff_path, avi_path, fourcc, fps):
    """
    Convert tiff file into avi file with the specified fourcc codec and fps
    The isColor parameter of the writer is harcoded set to False.

    Parameters:
    -----------
    tiff_path: str,
        Path to the tiff file
    avi_path: str
        Path to the output file
    fourcc: str, fourcc code
        0 means no coompression, other codecs will have some compression
        To learn more visit: https://www.fourcc.org/
    fps: float (should it be int?)
        Number of frames per second at which the recording was acquired

    To improve:
    ----------
    Write Multifile as option, so it can be set to True

    """

    #corrects fourcc nomenclature
    if fourcc == '0':
        fourcc=0
    else:
        fourcc=cv2.VideoWriter_fourcc(*fourcc)
    
    #make fps a float
    fps=float(fps)
    
    #tiff read object
    with tiff.TiffFile(tiff_path, multifile=False) as tif:
        #print(tif)
        frameSize=tif.pages[0].shape
        frame_height, frame_width=tif.pages[0].shape
        video_out = cv2.VideoWriter(avi_path, apiPreference=0, fourcc=fourcc, fps=fps, frameSize=(frame_width,frame_height), isColor=False)

        for i, page in enumerate(tif.pages):
            #print(i)
            img=page.asarray()
            #img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
            video_out.write(cv2.convertScaleAbs(img))#if img is uint16 it can't save it
            #if i>20: break
    video_out.release()

def ometiff2bigtiff(path):
    """
    List all ome.tiff in a directory and make them one bigtiff
    Somehow it gives an error for the last ome tiff, but resulting .btf is fine.

    IMPORTANT: This ometiff2big tiff removes the Z-Stack information in a recording with Z stacks!
    At least if the number of Z Stacks is inconsistent, which is the case for the current writer in ome.tiff. While recording, the microscope saves the ome.tiff file even before the z-stack is finished.
    
    Parameters:
    -----------
    path: str,
        Path to the directory containing the several ome tiff files.

    """
    print(path)
    if path.endswith('/'):
        output_filename=path+re.split('/',path)[-2]+'bigtiff.btf'
    else:
        output_filename=path+'/'+re.split('/',path)[-1]+'bigtiff.btf'
    with tiff.TiffWriter(output_filename, bigtiff=True) as output_tif:
        print(f'list of files is {os.listdir(path)}')
        for file in natsorted(os.listdir(path)):
            print(os.path.join(path,file))
            if file.endswith('ome.tif') and 'bg' not in file:
                print(os.path.join(path,file))
                with tiff.TiffFile(os.path.join(path,file), multifile=False) as tif:
                    for page in tif.pages:
                        img = page.asarray()
                        output_tif.write(img, photometric='minisblack', contiguous=True)#, description=omexmlMetadataString)


def ometiff2bigtiffZ(path, output_dir=None, actually_write=True, num_slices=None):
    """
    This function was copied from video_conversions/Python/bigtiff/
    """
    if output_dir is None:
        output_dir = path
    if path.endswith('/'):
        output_filename=output_dir+re.split('/',path)[-2]+'bigtiff.btf'
    else:
        output_filename=output_dir+'/'+re.split('/', path)[-1]+'bigtiff.btf'

    print(f"File will be written that is divisible by {num_slices}")
    print(f"And written to filename {output_filename}")
    total_num_frames = 0
    buffer = []
    with tiff.TiffWriter(output_filename, bigtiff=True) as output_tif:
        for i_file, file in enumerate(natsorted(os.listdir(path))):
            if not file.endswith('ome.tif') or 'bg' in file:
                continue
            this_ome_tiff = os.path.join(path,file)
            print("Currently reading: ")
            print(this_ome_tiff)
            with tiff.TiffFile(this_ome_tiff, multifile=False) as tif:
                for i, page in enumerate(tif.pages):
                    print(f'Page {i}/{len(tif.pages)} in file {i_file}')
                    # Bottleneck line
                    img = page.asarray()
                    # Convert to proper format, and write single frame
                    # img = (alpha*img).astype('uint8')
                    total_num_frames += 1
                    if num_slices is None:
                        if actually_write:
                            output_tif.write(img, photometric='minisblack')
                    else:
                        buffer.append(img)
                        if len(buffer) >= num_slices:
                            print(f"Writing {num_slices} frames from buffer...")
                            for img in buffer:
                                if actually_write:
                                    output_tif.write(img, photometric='minisblack', contiguous=True)
                            buffer = []
            if len(buffer)>0:
                print(f"{len(buffer)} frames not written")

                    # if num_frames is not None and i > num_frames: break

def max_projection_3d(input_filepath, output_filepath, fold_increase=3, nplanes=20, flip=False):

    """
    Create a visualization image of a volume, with the 3 max projections possible.
    To improve: Make another function that does the same but in a figure. Then the fold increase would not have to be int.
    Parameters:
    ------------
    input_filepath: str,

    output_filepath: str,

    fold_increase: int, (it can't be float, so the ratio dimensions xyz are not exactly real)
        Expands the z dimension so that the image has crrect dimensions. Depends on the ratio between xy pixel size and z-step size.
    nplanes: int,

    """
    with tiff.TiffWriter(output_filepath, bigtiff=True) as output_tif:
        with tiff.TiffFile(input_filepath) as tif:
            for idx, page in enumerate(tif.pages):
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
                    # rotates max0
                    max0=np.transpose(max0)
                    max1 = np.repeat(max1, fold_increase, axis=1)

                    #defines corner array dimensions and fill value of the corner matrix
                    fill_value=100
                    corner_matrix=np.full((fold_increase*img_stack.shape[2],fold_increase*img_stack.shape[2]),fill_value, dtype='uint16')
                    
                    #flip if needed (for Green Channel, since the image is mirrored compared to red channel)
                    if flip==True:
                        max2=cv2.flip(max2,1)
                        max0=cv2.flip(max0,1)
                    # concatenate the different max projections into one image

                    vert_conc_1 = cv2.hconcat([max2,max1])
                    vert_conc2 = cv2.hconcat([max0, corner_matrix])
                    final_img = cv2.vconcat([vert_conc_1,vert_conc2])
                    
                    #save the 3 max projection image
                    output_tif.write(final_img, photometric='minisblack', contiguous=True)

####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:


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
    Images have to be .tif, can't be PNG.
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
        #try to get metadata from imageJ
        try:
            files = tif.imagej_metadata['Info'].split('\n')
            metadata=True
        except:
            metadata=False
        for idx,page in enumerate(tif.pages):
            img=page.asarray()
            #if metadata is True: name according to metadata, else name image1.tif,etc.
            if metadata==True: filename=files[idx]
            else: filename='image'+str(idx)+'.tif'
            tiff.imwrite(os.path.join(output_path, filename), img)

def contours_length(img):
    """
    Return length and perimeter of the contours in an image.
    Length is assumed to be half of the perimeter. Only valid for elongated contours.

    Parameters
    -------------
    img: numpy_array,
    name of the input stack

    Returns
    ------------
    contours_len: numpy array,
    contains the length of the contours
    contours_peri: numpy array,
    contains the perimeter of the contours
  
    """
    _,cnts,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_peri=[]
    
    for cnt in cnts:
        contours_peri.append(cv2.arcLength(cnt, True))
    
    contours_peri=np.array(contours_peri)
    contours_len=contours_peri/2
    
    return  contours_len, contours_peri

def z_projection(img,projection_type):
    """
    Parameters:
    ------------
    img, 3-D numpy array
    Image stack that needs to be projected across the z dimension
    
    projection type, str
    String containing one of the 4 projections options: max, min, mean or median.
    
    Returns:
    ------------
    projected_img, numpy array
    Contains the projected img
    """
    if projection_type == 'max':
        projected_img = np.max(img, axis=0)
    if projection_type == 'min':
        projected_img = np.min(img, axis=0)
    if projection_type == 'mean':
        projected_img = np.mean(img, axis=0)
    if projection_type == 'median':
        projected_img = np.median(img, axis=0)

    return projected_img


def z_projection_parser(img_path, output_path, projection_type):
    img=tiff.imread(img_path)
    projected_img=z_projection(img, projection_type)
    tiff.imwrite(output_path,projected_img)
    return None






