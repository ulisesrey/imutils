import tifffile as tiff
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
import zarr
import napari


project_path = '/Volumes/scratch/neurobiology/zimmer/ulises/active_sensing/epifluorescence_recordings/25x/20220707_25x/data/ZIM1661_worm9'

red_img_path=glob.glob(os.path.join(project_path, '*/*worm*Ch0bigtiff.btf'))[0]
green_img_path=glob.glob(os.path.join(project_path, '*/*worm*Ch1bigtiff.btf'))[0]

print(red_img_path, green_img_path)

green_img=tiff.imread(green_img_path, aszarr=True)
green_img = zarr.open(green_img, mode='r')

red_img=tiff.imread(red_img_path, aszarr=True)
red_img = zarr.open(red_img, mode='r')


#with napari.gui_qt() as app:
viewer = napari.Viewer()

red_layer = viewer.add_image(red_img)
#red_layer.colormap = 'red'
green_layer = viewer.add_image(green_img)
napari.run()