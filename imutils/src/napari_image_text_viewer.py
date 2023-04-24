import tifffile as tiff
import zarr
import napari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Fetch project folders

#projects = glob.glob("/Volumes/scratch/neurobiology/zimmer/ulises/test_area/autoscope_snakemake/data/worm1/*/")[0]
projects = glob.glob("/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221123/data/*worm8/*BH*/")[0]
print(projects)

img_path_list=glob.glob(os.path.join(projects, "*worm_segmented_mask_coil_segmented_mask.btf"))
# add normalised image
#norm=glob.glob(os.path.join(projects, "*normalised.btf"))
#img_path_list.append(norm[0])
print(img_path_list)

# read csv file, ideally the reformatted one
beh_path = glob.glob(os.path.join(projects, "*turns_annotation.csv"))[0]
#merged_spline_data_path = "/Users/ulises.rey/local_data/test_spline/old2_skeleton_merged_spline_data.csv"
print(beh_path)

beh_df = pd.read_csv(beh_path, index_col=0)

#rolling average
#print("You are performing a rolling average! BE aware!")
#spline_df = spline_df.rolling(48, center=True, min_periods=24).mean()


# #get x and y coords in the correct format
points = (np.ones((beh_df.shape[0], 2)) * 100)


# get k curvature as a point property under confidence
point_properties = {
    'confidence': beh_df["turn"].to_numpy()
}

# OR USE THEXT PARAMETERS
# text_parameters = {
#     'string': 'label: {label}\ncirc: {circularity:.2f}',
#     'size': 12,
#     'color': 'green',
#     'anchor': 'upper_left',
#     'translation': [-3, 0],
# }


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
        # text = text_parameters,
        face_color='confidence',
        face_colormap='bwr',
        face_contrast_limits=[-1, 1],
        size=5
    )

napari.run()

print('end')

