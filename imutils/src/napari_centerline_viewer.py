import tifffile as tiff
import zarr
import napari
import pandas as pd
import numpy as np


# I think following these instructions it should be doable to read the csv and draw the points on top:
#https://napari.org/stable/howtos/layers/points.html
#Section : Setting edge or face color with a colormap¶

# read csv file, ideally the reformatted one
merged_spline_data_path = "/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BHbigtiff_skeleton_merged_spline_data.csv"
spline_df = pd.read_csv(merged_spline_data_path, header=[0,1], index_col=0)

x = spline_df.loc[0][:,'x'].values
y = spline_df.loc[0][:,'y'].values

points = np.column_stack((x,y))
point_properties = {
    'confidence': spline_df.loc[0][:,'k'].values,
}

#There is a notebook like this in the epifluorescence package
img_path_list=["/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BHbigtiff_AVG_background_subtracted_normalised_unet_segmented_weights_5358068_1_mask_coil_segmented_weights_5848336_0_mask.btf"]

# with napari.gui_qt() as app:
viewer = napari.Viewer()

for img_path in img_path_list:

    #image_layer
    img = tiff.imread(img_path, aszarr=True)
    img = zarr.open(img, mode='r')
    viewer.add_image(img, blending='additive')

    #points layer
    points_layer = viewer.add_points(
        points,
        properties=point_properties,
        face_color='confidence',
        face_colormap='viridis',
    )

napari.run()

