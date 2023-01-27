import tifffile as tiff
import zarr
import napari
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# I think following these instructions it should be doable to read the csv and draw the points on top:
#https://napari.org/stable/howtos/layers/points.html
#Section : Setting edge or face color with a colormap¶


#With this tutorial it coul be done that it updates across time
# https://napari.org/stable/gallery/points-over-time.html

# read csv file, ideally the reformatted one
merged_spline_data_path = "/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BHbigtiff_skeleton_merged_spline_data.csv"
spline_df = pd.read_csv(merged_spline_data_path, header=[0,1], index_col=0)

spline_df = spline_df.rolling(48, center=True, min_periods=24).mean()


# df.loc[0]['0'][["x", "y"]]
#x = spline_df.xs(('x',), level=('coords',), axis=1)#spline_df.loc[frame][:,'x'].values
#y = spline_df.xs(('y',), level=('coords',), axis=1)#spline_df.loc[frame][:,'y'].valuesy

points = (
    spline_df.rename_axis("index")
    .stack("segment")
    .reset_index()[["index", "x", "y"]]
    .to_numpy()
)



#reshape data to be 1D
#points = points.reshape(-1) #points, newshape=)
print(points.shape)

# points = np.hstack([t_vec, points])
#print(points.shape)

# remove dummy starter point?
#points = points[1:, :]
#print(points.shape)

#old methd

#good method
point_properties = {
    'confidence': spline_df.rename_axis("index").stack("segment").reset_index()[["index", "k"]].to_numpy()[:,1]
}

#this brings nans to 0, # Fill NaNs with 0 so that the color can be plotted, otherwise error
# point_properties = np.nan_to_num(point_properties)



#There is a notebook like this in the epifluorescence package
img_path_list=["/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BHbigtiff_AVG_background_subtracted_normalised_unet_segmented_weights_5358068_1_mask_coil_segmented_mask.btf"]

# with napari.gui_qt() as app:
viewer = napari.Viewer()

for img_path in img_path_list:

    #image_layer
    img = tiff.imread(img_path, aszarr=True)
    img = zarr.open(img, mode='r')
    viewer.add_image(img, blending='additive')

    #points layer
    #points_layer = viewer.add_points(points)
    points_layer = viewer.add_points(
        data=points,
        properties=point_properties,
        face_color='confidence',
        face_colormap='bwr',
        size=10
    )

napari.run()

print('end')

