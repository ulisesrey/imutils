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
projects = glob.glob("/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/*worm1*/*BH*/")[0]
print(projects)

img_path_list=glob.glob(os.path.join(projects, "*raw_stack_AVG_background_subtracted.btf")) #"*worm_segmented_mask_coil_segmented_mask.btf"))
# norm = glob.glob(os.path.join(projects, "*normalised.btf"))
# img_path_list.append(norm[0])
print(img_path_list)

# read csv file, ideally the reformatted one
merged_spline_data_path = glob.glob(os.path.join(projects, "*skeleton_merged_spline_data_avg.csv"))[0]
#merged_spline_data_path = "/Users/ulises.rey/local_data/test_spline/old2_skeleton_merged_spline_data.csv"
print(merged_spline_data_path)

spline_df = pd.read_csv(merged_spline_data_path, header=[0,1], index_col=0)
#rolling average
#print("You are performing a rolling average! BE aware!")
#spline_df = spline_df.rolling(48, center=True, min_periods=24).mean()


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

#img_path = img_path_list[0]
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
        size=5
    )

napari.run()

print('end')

