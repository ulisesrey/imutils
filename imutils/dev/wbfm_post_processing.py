import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from skimage import filters

import tifffile as tiff

import cv2

#scripts to run on the cluster for quantifying bleaching

#input_filename='/Volumes/scratch/ulises/wbfm/20211210/data/worm1/2021-12-10_11-59-29_ZIM2156_worm1-channel-0-pco_camera1/2021-12-10_11-59-29_ZIM2156_worm1-channel-0-pco_camera1bigtiff_z_project.btf'
#output_filename='/Volumes/scratch/ulises/wbfm/20211210/data/worm1/scarlet_mask/2021-12-10_11-59-29_ZIM2156_worm1-channel-0-pco_camera1bigtiff.btf'
# img=tiff.imread(input_filename)
# plt.imshow(img)

# create csv objects
# csvfile_corrected_head = open(csv_output_path + '_skeleton_corrected_head_coords.csv', 'w', newline='')
# csv_writer_head = csv.writer(csvfile_corrected_head)

# df=pd.DataFrame()
#
# with tiff.TiffFile(input_filename) as tif, tiff.TiffWriter(output_filename) as tif_writer:
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
#     #     csv_writer_head.writerow([idx, mean, min, max])
#     # csvfile_corrected_head.close()

def create_mask(input_filepath, output_filepath):
    """"
    function to create a mask out of the z_3D_projection
    """
    with tiff.TiffFile(input_filepath) as tif, tiff.TiffWriter(output_filepath) as tif_writer:
        for idx, page in enumerate(tif.pages):
            print('Number of pages: ', len(tif.pages))
            img = page.asarray()
            img = img[:650, :900]
            blurred_img = filters.gaussian(img, 5)
            ret, mask = cv2.threshold(blurred_img, 0.003, 255, cv2.THRESH_BINARY)
            mask=mask.astype(bool)
            tif_writer.write(mask, contiguous=True)

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
            mask=tif_mask.pages[idx].asarray()
            mask = np.bool_(mask)
            print('mean is ', np.mean(img[mask]))
            df.loc[idx,'area'] = len(img[mask])
            df.loc[idx,'mean'] = np.mean(img[mask])
            df.loc[idx,'min'] = np.min(img[mask])
            df.loc[idx,'max'] = np.max(img[mask])
    df.to_csv(csv_output_filepath)

def plot_bleaching_curve():
    path='/Volumes/scratch/ulises/wbfm/20211210/data/worm2/Results_scarlett.csv'
    df=pd.read_csv(path)#, sep='\t')
    print(df.head(10))
    print(list(df.columns))
    fig,(ax1,ax2)=plt.subplots(nrows=2)
    df['Mean'].plot(ax=ax1)
    ax1.set_ylabel('Mean Pixel Intensity')

    ax1.set_ylim([100,200])
    df['Max'].plot(ax=ax2)
    ax2.set_ylabel('Max Pixel Intensity')
    ax2.set_ylim([100,600])
    plt.show()