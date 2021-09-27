import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from imutils.src import imfunctions
from pathlib import Path
import cv2


def calculate_signal(rfp_filepath,rfp_mask_filepath,gcamp_filepath,background_gfp,background_red):
    """
    returns a list of background corrected gcamp and rfp signal from a roi specified via the mask
    Parameters:
    rfp_filepath: btf file, rfp signal
    rfp_mask: btf file,
    gcamp_filepath: gfp signal
    background: background intensity
    """
    gcamp_signal=[]
    rfp_signal=[]
    with tiff.TiffFile(rfp_mask_filepath, multifile=True) as tif_mask,\
        tiff.TiffFile(gcamp_filepath, multifile=False) as tif_gcamp,\
        tiff.TiffFile(rfp_filepath, multifile=False) as tif_rfp:
        for i, page in enumerate(tif_mask.pages):
            if i>len(tif_gcamp.pages)-1:break
            mask=page.asarray()
            gcamp=tif_gcamp.pages[i].asarray()
            rfp=tif_rfp.pages[i].asarray()
        
            #find where the mask is
            roi=np.where(mask==255)
        
            gcamp_roi_signal=np.mean(gcamp[roi])
            rfp_roi_signal=np.mean(rfp[roi])
                   
            #append
            gcamp_signal.append(gcamp_roi_signal)
            rfp_signal.append(rfp_roi_signal)  
        gcamp_signal=np.array(gcamp_signal)
        rfp_signal=np.array(rfp_signal)
        gcamp_signal=gcamp_signal-background_gfp
        rfp_signal=rfp_signal-background_red
    return gcamp_signal,rfp_signal 


def plot_gcamp_rfp_ratio(gcamp_signal,rfp_signal):
    fig, axes=plt.subplots(figsize=(20,5),nrows=3, dpi=150)
    axes[0].plot(gcamp_signal)
    axes[0].set_ylabel('GCamp',fontsize=15)
    axes[1].plot(rfp_signal)
    axes[1].set_ylabel('RFP',fontsize=15)
    axes[2].plot(gcamp_signal/rfp_signal)
    axes[2].set_ylabel('GCamp/RFP',fontsize=15) 
    return axes
    
    
    
def make_contour_based_binary_all_files(input_path,output_path,median_blur, lt, ht, contour_size, tolerance, area_to_fill):
    """
    applies the make_contour_based_binary_function to all btf files in a directory
    Parameters:
    --------------------------
    median_blur, lt, ht, contour_size, tolerance, area_to_fill: parameters of make_contour_based bianry
    input_path: str, directory containing btfs
    output_path:,str btfs are saved as (filename)_binary.btf in the output directory
    """
    #convert input path fron string to path object
    input_path=Path(input_path)
    
    #grab all btf files from the input directory
    for file in input_path.glob('*.btf'):
        
        #specifiy utput path and new filename for each file
        output_path=f'{output_path}/{file.stem}_binary.btf'
        imfunctions.make_contour_based_binary(file, output_path, median_blur, lt, ht, contour_size, tolerance, area_to_fill)
        
        
        
def binarize_btf(input_path,output_path,median_blur,lower_threshold,higher_threshold):
    
    """
    binarizes btf files
    """
    
    
    #create reader and writer objects
    with tiff.TiffWriter(output_path, bigtiff=True) as tif_writer:
        with tiff.TiffFile(input_path, multifile=False) as tif:
        
            #read in pages
            for i, page in enumerate(tif.pages):
                img=page.asarray()
            
                #apply median Blur
                if median_blur!=0:
                    img=cv2.medianBlur(img,median_blur)

                #apply threshold
                ret, new_img = cv2.threshold(img,lower_threshold,higher_threshold,cv2.THRESH_BINARY)
                tif_writer.save(new_img,contiguous=True)
                

def get_new_file_names(input_file,output_path,add_to_name,file_extension):
    """
    Parameters:
    returns a string containing an output path and a new filename.
    new filename consits of the stem name of the input, a specified  addition to the name and extension
    ----------------
    input_file: Path object
    output_path:path were the new files should be saved
    add_to_name:str,string to add to end of the new filename (for example: _binary)
    file_extension:format of the new file (for example: .btf)
    
    """
    
    #specifiy utput path and new filename for each file
    new_filename=f'{output_path}/{input_file.stem}{add_to_name}{file_extension}'
    
    return new_filename




def calculate_background_intensity(rfp_mask_filepath,rfp_filepath,gcamp_filepath):
    
    """
    returns a list of background corrected gcamp and rfp signal from a roi specified via the mask
    
    Parameters:
    -----------------------------------
    rfp_filepath: btf file, rfp signal
    rfp_mask: btf file,
    gcamp_filepath: gfp signal
    background: background intensity
    """

    gcamp_background=[]
    rfp_background=[]

    with tiff.TiffFile(rfp_mask_filepath, multifile=True) as tif_mask,\
        tiff.TiffFile(gcamp_filepath, multifile=False) as tif_gcamp,\
        tiff.TiffFile(rfp_filepath, multifile=False) as tif_rfp:
    
        for i, page in enumerate(tif_mask.pages):
            
            steps=len(tif_mask.pages)/10
            
            if i %steps==0:
        
        
                mask=page.asarray()
                gcamp=tif_gcamp.pages[i].asarray()
                rfp=tif_rfp.pages[i].asarray()
        
            
                #find where the mask is
                background=np.where(mask!=255)
        
                gcamp_background_single_frame=np.mean(gcamp[background])
                rfp_background_single_frame=np.mean(rfp[background])
        
                gcamp_background.append(gcamp_background_single_frame)
                rfp_background.append(rfp_background_single_frame)

        gcamp_background_mean=np.mean(np.array(gcamp_background))
        rfp_background_mean=np.mean(np.array(rfp_background))
        
    return (gcamp_background_mean,rfp_background_mean)
