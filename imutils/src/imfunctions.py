import cv2
import tifffile as tiff

import numpy as np
import matplotlib.pyplot as plt
import os
from natsort import natsorted
import re

import pandas as pd
import csv

from imutils.src.model import *

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


def stack_substract_background(input_filepath, output_filepath, background_img_filepath):
    """
    Substract the background image from a btf stack
    Parameters:
    ----------
    input_filepath, str
    input path to the tiff file
    output_filepath, str
    path to where the file will be written
    background_img_filepath, str
    path to the background image
    Returns:
    ----------
    """
    #load background image
    bg_img=tiff.imread(background_img_filepath)

    with tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
        with tiff.TiffFile(input_filepath, multifile=False) as tif:
            for i, page in enumerate(tif.pages):
                img=page.asarray()
                inv_img=cv2.bitwise_not(img)
                new_img=cv2.subtract(inv_img,bg_img)
                tif_writer.write(new_img, contiguous=True)

def make_contour_based_binary(stack_input_filepath, stack_output_filepath, median_blur, lower_threshold, higher_threshold, contour_size, tolerance, inner_contour_area_to_fill):
    """
    Produce a binary image based on contour and inner contour sizes.
    better than the make_binary before which was on centerline package
    Parameters:
    -----------
    A lot, too many?

    Returns:
    --------
    """
    with tiff.TiffWriter(stack_output_filepath, bigtiff=True) as tif_writer:
        with tiff.TiffFile(stack_input_filepath, multifile=False) as tif:
            for i, page in enumerate(tif.pages):
                #loads the first frame and inverts it
                img=page.asarray()
                #median Blur
                img=cv2.medianBlur(img,median_blur)
               
                #apply threshold
                ret, new_img = cv2.threshold(img,lower_threshold,higher_threshold,cv2.THRESH_BINARY)
                #draw_some_contours does not need imfunctions.draw_some_contours in here. But outside this file.
                worm_contour_img=draw_some_contours(new_img,contour_size=contour_size,tolerance=tolerance, inner_contour_area_to_fill=inner_contour_area_to_fill)
            
                tif_writer.write(worm_contour_img, contiguous=True)



def unet_segmentation_contours_with_children(input_filepath, output_filepath, weights_path):
    """
    Run through the unet segmentation the contours with children.
    TO DO: It would be more efficient to do a list of frames.
    Parameters:
    -----------
    input_filepath, str
    output_filepath, str
    weights_path, str
    Returns:
    -----------

    """

    model=unet()
    model.load_weights(weights_path)

    with tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
            with tiff.TiffFile(input_filepath, multifile=False) as tif:
                for i, page in enumerate(tif.pages):
                    img=page.asarray()
                    
                    #find contours with children
                    contours_with_children=extract_contours_with_children(img)

                    #make a copy of the original image here in order to paste more than one contour with children
                    new_img=img.copy()
                    for cnt_idx,cnt in enumerate(contours_with_children):
                            x,y,w,h = cv2.boundingRect(cnt)
                            #make the crop
                            cnt_img=img[y:y+h,x:x+w]

                            #run U-Net network:
                            cnt_img=cv2.resize(cnt_img, (256,256))
                            cnt_img=np.reshape(cnt_img,cnt_img.shape+(1,))
                            cnt_img=np.reshape(cnt_img,(1,)+cnt_img.shape)

                            #normalize to 1 by dividing by 255
                            cnt_img=cnt_img/255
                            results = model.predict(cnt_img)
                            #reshape results
                            results_reshaped=results.reshape(256,256)
                            #resize results
                            results_reshaped=cv2.resize(results_reshaped, (w,h))
                            #multiply it by 255
                            results_reshaped=results_reshaped*255

                            #paste it into the original image
                            new_img[y:y+h,x:x+w]=results_reshaped

                    tif_writer.write(new_img, contiguous=True)

####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
####### THE FUNCTION BELOW CAN'T BE CALLED FROM THE IMUTILS PARSER YET:
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
   
    """
    parser do run the z_projection function
    Parameters:
    ----------
    img_path, str
    output_path, str
    projection_type, str
    Returns:
    ----------
    Writes the projection. Function itself returns None
    """
    img=tiff.imread(img_path)
    projected_img=z_projection(img, projection_type)
    tiff.imwrite(output_path,projected_img)
    return None

def draw_some_contours(img,contour_size,tolerance,inner_contour_area_to_fill):
    """
    Return img with drawn contours based on size, filling contours below inner_contour_area_to_fill
    Parameters:
    -----------
    img, numpy array
    image from where the contours will be taken
    contour_size, float
        expected area of the contour to be extracted
    tolerance, float
        tolerance around which other contours will be accepted. e.g. contour_size 100 and tolerance 0.1 will include contours from 90 to 110.
    inner_contour_area_to_fill, float
        area of inner contours that will be filled
        
    Returns:
    -----------
    img_contours, numpy array
        image with drawn contours
    """

    #get contours
    _,cnts,hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    #good contours index
    cnts_idx=[]#np.array([])
    #create empty image
    img_contours=np.zeros(img.shape)

    for cnt_idx, cnt in enumerate(cnts):
        cnt_area=cv2.contourArea(cnt)
        #if the contour area is between the expected values with tolerance, save contour in cnts_idx and draw it
        if (contour_size*(1-tolerance) <cnt_area<contour_size*(1+tolerance)):
            cnts_idx.append(np.array(cnt_idx))
            cv2.drawContours(img_contours,cnts, cnt_idx, color=255, thickness=-1, hierarchy=hierarchy, maxLevel=1)
        
        #if the current cnt_idx has as a parent a contour in good countours (cnts_idx)
        if hierarchy[0][cnt_idx][3] in cnts_idx:
            #(and) if it is smaller than inner contour, draw it
            if cnt_area<inner_contour_area_to_fill:
            #print(cv2.contourArea(contours[j]))
                cv2.drawContours(img_contours,cnts, cnt_idx, color=255, thickness=-1)
    
    #convert the resulting image into a 8 binary numpy array
    img_contours=np.array(img_contours, dtype=np.uint8)

    return img_contours
    
def extract_contours_with_children(img):
    """
    Find the contours that have a children in the given image and return them as list
    
    Parameters:
    -----------
    img, np.array
    Returns:
    -----------
    contours that have a children, list of contours with children
    Important does not return the children, only the contour that has children.
    """
    _, cnts, hierarchy=cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #print(len(cnts))
    #print(hierarchy)
    contours_with_children=[]
    for cnt_idx, cnt in enumerate(cnts):
    # draw contours with children: last column in the array is -1 if an external contour, column 2 is different than -1 meaning it has children
        if hierarchy[0][cnt_idx][3] == -1 and hierarchy[0][cnt_idx][2]!=-1:
            contours_with_children.append(cnt)
            # not needed, do it outside this function
            #get coords of boundingRect
            #x,y,w,h = cv2.boundingRect(cnt)
            #make the crop
            #cnt_img=img[y:y+h,x:x+w]
    return contours_with_children

