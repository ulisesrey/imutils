import tifffile as tiff
import zarr
import napari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Fetch project folders

projects = glob.glob("/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221123/data/*worm1*/*BH*/")[0]
print(projects)

img_path_list=glob.glob(os.path.join(projects, "*background_subtracted_normalised.btf"))
print(img_path_list)

# read csv file, ideally the reformatted one
merged_spline_data_path = glob.glob(os.path.join(projects, "*skeleton_merged_spline_data.csv"))[0]
print(merged_spline_data_path)

spline_df = pd.read_csv(merged_spline_data_path, header=[0,1], index_col=0)
#rolling average
spline_df = spline_df.rolling(48, center=True, min_periods=24).mean()

#get x and y coords in the correct format
points = (
    spline_df.rename_axis("index")
    .stack("segment")
    .reset_index()[["index", "x", "y"]]
    .to_numpy()
)


# get k curvature as a point property under confidence
point_properties = {
    'confidence': spline_df.rename_axis("index").stack("segment").reset_index()[["index", "k"]].to_numpy()[:,1]
}



# with napari.gui_qt() as app:
viewer = napari.Viewer()

for img_path in img_path_list:

    #image_layer
    img = tiff.imread(img_path, aszarr=True)
    img = zarr.open(img, mode='r')
    viewer.add_image(img, blending='additive')

    #points layer
    points_layer = viewer.add_points(
        data=points,
        properties=point_properties,
        face_color='confidence',
        face_colormap='bwr',
        face_contrast_limits=[-0.02, 0.02],
        size=2
    )

napari.run()

print('end')

