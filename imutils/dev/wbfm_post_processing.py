import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage import filters

import tifffile as tiff

import cv2

import glob

#scripts to run on the cluster for quantifying bleaching
#
# input_filepath='/Volumes/scratch/ulises/wbfm/20211210/data/worm3/2021-12-10_14-33-15_ZIM2156_worm3-channel-1-pco_camera2/2021-12-10_14-33-15_ZIM2156_worm3-channel-1-pco_camera2bigtiff_z_project.btf'
# output_filepath='/Volumes/scratch/ulises/wbfm/20211210/data/worm1/scarlet_mask/2021-12-10_11-59-29_ZIM2156_worm1-channel-0-pco_camera1bigtiff.btf'
# #
# # # img=tiff.imread(input_filename)
# # # plt.imshow(img)
# # #
# # # create csv objects
# # # csvfile_corrected_head = open(csv_output_path + '_skeleton_corrected_head_coords.csv', 'w', newline='')
# # # csv_writer_head = csv.writer(csvfile_corrected_head)
# #
# df=pd.DataFrame()
#
# with tiff.TiffFile(input_filepath) as tif, tiff.TiffWriter(output_filepath) as tif_writer:
#     for idx, page in enumerate(tif.pages):
#         img=page.asarray()
#         img=img[:650,:900]
#         blurred_img=filters.gaussian(img, 5)
#         ret, mask = cv2.threshold(blurred_img, 0.003, 255, cv2.THRESH_BINARY)
#         tif_writer.write(mask, contiguous=True)
#         #convert the mask to boolean
#         mask = np.bool_(mask)
#         print('mean is ', np.mean(img[mask]))
#         df.loc[idx,'area'] = len(img[mask])
#         df.loc[idx,'mean'] = np.mean(img[mask])
#         df.loc[idx,'min'] = np.min(img[mask])
#         df.loc[idx,'max'] = np.max(img[mask])
#         if idx>10: break
#     df.to_csv('/Volumes/scratch/ulises/wbfm/20211210/data/worm1/Results_python.csv')
    #     csv_writer_head.writerow([idx, mean, min, max])
    # csvfile_corrected_head.close()

def create_mask(input_filepath, output_filepath):
    """"
    function to create a mask out of the z_3D_projection
    """
    with tiff.TiffFile(input_filepath) as tif, tiff.TiffWriter(output_filepath) as tif_writer:
        print('Number of pages: ', len(tif.pages))
        for idx, page in enumerate(tif.pages):
            img = page.asarray()
            img = img[:650, :900]
            blurred_img = filters.gaussian(img, 5)
            #ret, mask = cv2.threshold(blurred_img, 0.003, 10000, cv2.THRESH_BINARY)
            # if blurred_img is higher than 0.03, write 1, if not 0.
            thresh=filters.threshold_otsu(blurred_img)
            # mask=img>thresh
            mask=np.where(blurred_img>thresh, 255, 0)
            mask=np.array(mask, dtype=np.uint8)
            tif_writer.write(mask)
            #if idx>400: break

# output_filepath='/Volumes/scratch/ulises/wbfm/20211210/2021-12-10_11-59-29_ZIM2156_worm1-channel-0-pco_camera1bigtiff.btf'
#
# create_mask(input_filepath, output_filepath)

def quantify_mask(input_filepath, mask_filepath, csv_output_filepath):
    """
    Args:
        input_filepath: str
        btf with the max projection or image where you want to calculate the values in the mask
        mask_filepath: str
        btf with the mask
        csv_output_filepath: str
        output where the dataframe will be saved
    Returns:

    """
    df=pd.DataFrame()
    with tiff.TiffFile(input_filepath) as tif, tiff.TiffFile(mask_filepath) as tif_mask:
        for idx, page in enumerate(tif.pages):
            img=page.asarray()
            img = img[:650, :900]
            mask=tif_mask.pages[idx].asarray()
            mask = np.bool_(mask)
            print('mean is ', np.mean(img[mask]))
            df.loc[idx,'area'] = len(img[mask])
            df.loc[idx,'mean'] = np.mean(img[mask])
            df.loc[idx,'min'] = np.min(img[mask])
            df.loc[idx,'max'] = np.max(img[mask])
            df.loc[idx,'10th_percentile']=np.percentile(img[mask], 10)
            df.loc[idx, '10th_percentile_mean'] = np.mean(img[mask]>np.percentile(img[mask], 10))
    df.to_csv(csv_output_filepath)

def plot_bleaching_curve(project_path, channel):
    """
    Args:
        project_path: str,
         e.g. /scratch/ulises/wbfm/20211210/data/worm3/
        channel: str,
        e.g. 'gcamp' or 'scarlett'

    Returns:
    ax
    """
    path=os.path.join(project_path,'Results_'+channel+'.csv')
    df=pd.read_csv(path)#, sep='\t')

    fig, axes = plt.subplots(nrows=2)
    df['mean'].plot(ax=axes[0])
    axes[0].set_ylabel('Mean Pixel Intensity')
    axes[0].set_ylim([100,1200])

    df['max'].plot(ax=axes[1])
    axes[1].set_ylabel('Max Pixel Intensity')
    axes[1].set_xlabel('Time (volumes)')
    #axes[1].set_ylim([100,4000])
    return fig, axes

def plot_max_bleaching_curve(project_path, channel):

    path=os.path.join(project_path,'Results_'+channel+'.csv')
    df=pd.read_csv(path)#, sep='\t')

    fig, axes = plt.subplots(figsize=(10,4))
    df['max'].plot(ax=axes)
    axes.set_ylabel('Max Pixel Intensity')
    axes.set_xlabel('Time (volumes)')
    #axes[1].set_ylim([100,4000])
    return fig, axes

# projects=glob.glob('/Volumes/scratch/ulises/wbfm/20211210/data/worm*')
# channels=['gcamp', 'scarlet']
#
# for project_path in projects:
#     print(project_path)
# #project_path='/Volumes/scratch/ulises/wbfm/20211210/data/worm3/'
#     for channel in channels:
#         fig, axes = plot_max_bleaching_curve(project_path,channel)
#         title=project_path[-19:]+'_'+channel
#         fig.suptitle(title, fontsize=16)
#         plt.savefig(os.path.join(project_path,'Results_MAX'+channel), dpi=100)
# plt.show()