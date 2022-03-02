import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage import filters

import tifffile as tiff

import cv2

import glob


# scripts to run on the cluster for quantifying bleaching
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
    with tiff.TiffFile(input_filepath) as tif, tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
        print('Number of pages: ', len(tif.pages))
        for idx, page in enumerate(tif.pages):
            img = page.asarray()
            # TODO : Remove this part below and make the function of the input image
            img = img[:650, :900]
            blurred_img = filters.gaussian(img, 5)
            # ret, mask = cv2.threshold(blurred_img, 0.003, 10000, cv2.THRESH_BINARY)
            # if blurred_img is higher than 0.03, write 1, if not 0.
            thresh = filters.threshold_otsu(blurred_img)
            # mask=img>thresh
            mask = np.where(blurred_img > thresh, 255, 0)
            mask = np.array(mask, dtype=np.uint8)
            tif_writer.write(mask)
            # if idx>400: break


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
    df = pd.DataFrame()
    with tiff.TiffFile(input_filepath) as tif, tiff.TiffFile(mask_filepath) as tif_mask:
        for idx, page in enumerate(tif.pages):
            img = page.asarray()
            img = img[:650, :900]
            mask = tif_mask.pages[idx].asarray()
            mask = np.bool_(mask)
            # print('mean is ', np.mean(img[mask]))
            df.loc[idx, 'area'] = len(img[mask])
            df.loc[idx, 'mean'] = np.mean(img[mask])
            df.loc[idx, 'min'] = np.min(img[mask])
            df.loc[idx, 'max'] = np.max(img[mask])
            df.loc[idx, '10th_percentile'] = np.percentile(img[mask], 90)
            df.loc[idx, '10th_percentile_mean'] = np.mean(img[img > np.percentile(img[mask], 90)])
    df.to_csv(csv_output_filepath)


# def quantify_signal(df, column, time):
#     signal = df[column, time]
#     return signal
#

# csv_filepath = '/Volumes/scratch/ulises/wbfm/20211210/data/worm1/Results_scarlet.csv'
# column = '10th_percentile_mean'
# df = pd.read_csv(csv_filepath)
# df[column]


# input_filepath='/Volumes/scratch/ulises/wbfm/20211210/data/worm7/2021-12-10_17-09-46_ZIM2156_worm7-channel-0-pco_camera1/2021-12-10_17-09-46_ZIM2156_worm7-channel-0-pco_camera1bigtiff_z_project.btf'
# mask_filepath='/Volumes/scratch/ulises/wbfm/20211210/data/worm7/2021-12-10_17-09-46_ZIM2156_worm7-channel-0-pco_camera1/2021-12-10_17-09-46_ZIM2156_worm7-channel-0-pco_camera1bigtiff_z_project_mask.btf'
# csv_output_filepath='/Volumes/scratch/ulises/wbfm/20211210/test.csv'
# quantify_mask(input_filepath, mask_filepath, csv_output_filepath)


# def plot_bleaching_curve(project_path, channel):
#     """
#     Args:
#         project_path: str,
#          e.g. /scratch/ulises/wbfm/20211210/data/worm3/
#         channel: str,
#         e.g. 'gcamp' or 'scarlett'
#
#     Returns:
#     ax
#     """
#     path=os.path.join(project_path,'Results_'+channel+'.csv')
#     df=pd.read_csv(path)#, sep='\t')
#
#     fig, axes = plt.subplots(nrows=2)
#     df['mean'].plot(ax=axes[0])
#     axes[0].set_ylabel('Mean Pixel Intensity')
#     axes[0].set_ylim([100,1200])
#
#     df['max'].plot(ax=axes[1])
#     axes[1].set_ylabel('Max Pixel Intensity')
#     axes[1].set_xlabel('Time (volumes)')
#     #axes[1].set_ylim([100,4000])
#     return fig, axes

def plot_bleaching_curve(project_path, column, ax=None):
    #path = os.path.join(project_path, 'Results_' + channel + '.csv')
    df = pd.read_csv(project_path)  # , sep='\t')
    if ax == None:
        _, ax = plt.subplots(figsize=(10, 4))
    df[column].plot(ax=ax)
    ax.set_ylabel('Pixel Intensity')
    ax.set_xlabel('Time (volumes)')
    # axes[1].set_ylim([100,4000])
    return ax

# def bleaching_as_function_of_laser_power():
#     df['Date'] =
#     df['worm'] =
#     df['488_power(uW)'] =
#     df['561_power(uW)'] =
#     df['Starting_Intensity'] =
#     df['Ending_Intensity'] =
#     df['Bleaching'] =
#
#     return df



# channels=['gcamp', 'scarlet']
# for channel in channels:
#     files=glob.glob('/Volumes/scratch/ulises/wbfm/2021*/data/worm*/Results_'+str(channel)+'.csv')
#     column='10th_percentile_mean'
#     for file in files:
#         print(file)
#         fig, ax = plt.subplots(figsize=(10, 4))
#         ax=plot_bleaching_curve(file, column, ax=ax)
#         title=file[29:48]+' '+channel+' '+column
#         fig.suptitle(title, fontsize=16)
#         if channel == 'gcamp': ax.set_ylim([100,300])
#         if channel == 'scarlet': ax.set_ylim([150, 3500])
#         plt.savefig(os.path.splitext(file)[0]+str(column)+'.png', dpi=100)
# plt.show()


recordings=glob.glob('/Volumes/scratch/ulises/wbfm/2021*/data/worm*')

#
# for recording in recordings:
#     yaml_filepath=os.path.join(recording, 'config.yaml')
#     yaml_dict=yaml.safe_load(open(yaml_filepath))
#     df = pd.read_csv(os.path.join(recording, 'Results_scarlet.csv'))
#     if yaml_dict['recording_length_minutes']<15: continue
#     if yaml_dict['laser_561']<301:
#         print(recording)
#         #print intensity before and after
#         #print(df['10th_percentile_mean'].head(500).median(),df['10th_percentile_mean'].tail(500).median())
#         #print percentage bleaching
#         print((df['10th_percentile_mean'].head(500).median()-df['10th_percentile_mean'].tail(500).median())/df['10th_percentile_mean'].head(500).median())
#         df['10th_percentile_mean'].head(200).median()
#         df['10th_percentile_mean'].tail(200).median()
#         #df['10th_percentile_mean'].plot(color='r', alpha=0.5)
#     # if yaml_dict['laser_488'] < 601:
#     #     print('low:', df['10th_percentile_mean'].head(200).median(), df['10th_percentile_mean'].tail(200).median())
#     #     df['10th_percentile_mean'].head(200).median()
#     #     df['10th_percentile_mean'].tail(200).median()
#         df['10th_percentile_mean'].plot(color='k', alpha=0.5, title=recording)
#         plt.show()



#make figure of mask
#TODO write a function that merges two tiff files with alpha parameter
path_mask='/Volumes/scratch/ulises/wbfm/2021-12-10_14-33-15_ZIM2156_worm3-channel-0-pco_camera1bigtiff_z_project_mask-1.tif'
path_img='/Volumes/scratch/ulises/wbfm/2021-12-10_14-33-15_ZIM2156_worm3-channel-1-pco_camera2bigtiff_z_project-2.tif'

img=tiff.imread(path_img)
mask=tiff.imread(path_mask)

for i, frame in enumerate(img):
    fig, ax= plt.subplots()
    ax.imshow(frame)
    ax.imshow(mask[i], cmap='gray', alpha=0.25)
    ax.set_axis_off()
    plt.savefig('22021-12-10_14-33-15_ZIM2156_worm3-channel-1-pco_camera2bigtiff_z_project_' + str(i) + '.png', dpi=100)
    #break
