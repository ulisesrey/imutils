import tifffile as tiff
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import zarr
import napari

#There is a notebook like this in the epifluorescence package

project_path = '/Volumes/scratch/neurobiology/zimmer/active_sensing/zim06/20230111/data/worm1'
    #'/Volumes/scratch/neurobiology/zimmer/ulises/active_sensing/epifluorescence_recordings/20220520/data/ZIM1661_worm5/'

#beh_img_path=glob.glob(os.path.join(project_path, '*/*worm*Ch0-BHbigtiff.btf'))[0]
red_img_path=glob.glob(os.path.join(project_path, '*Ch0/*bigtiff*masked.btf'))[0]
green_img_path=glob.glob(os.path.join(project_path, '*Ch1/*bigtiff*masked.btf'))[0]

#two images:
#red_img_path = "/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221210/data/ZIM2165_Gcamp7b_worm8/2022-12-10_20-13_ZIM2165_worm8_Ch0-BH/2022-12-10_20-13_ZIM2165_worm8_Ch0-BHbigtiff_AVG_background_subtracted_normalised_worm_with_centerline.btf"
#green_img_path = "/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221210/data/ZIM2165_Gcamp7b_worm8/2022-12-10_20-13_ZIM2165_worm8_Ch0-BH/2022-12-10_20-13_ZIM2165_worm8_Ch0-BHbigtiff_AVG_background_subtracted_normalised_unet_segmented_weights_5358068_1_mask_coil_segmented_weights_5374562_0_mask.btf"
print(red_img_path, green_img_path)

#beh_img=tiff.imread(beh_img_path, aszarr=True)
#beh_img = zarr.open(beh_img, mode='r')

green_img=tiff.imread(green_img_path, aszarr=True)
green_img = zarr.open(green_img, mode='r')

red_img=tiff.imread(red_img_path, aszarr=True)
red_img = zarr.open(red_img, mode='r')


#with napari.gui_qt() as app:
viewer = napari.Viewer()

red_layer = viewer.add_image(red_img, blending='additive')
red_layer.colormap = 'red'

green_layer = viewer.add_image(green_img, blending='additive')
green_layer.colormap = 'green'
#beh_layer=viewer.add_image(beh_img)

napari.run()

