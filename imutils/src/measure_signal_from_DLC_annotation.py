# based on the notebook with the same name

import pandas as pd
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import tifffile as tiff

# define the actual function
def measure_signal_from_coords(image_path, coords, dist):
    n_values = []
    total_intensity = []
    # TODO:  add max intensity

    with tiff.TiffFile(image_path) as tif:
        for i, page in enumerate(tif.pages):
            #print(i)
            img = page.asarray()
            x, y = int(round(coords[i][0])), int(round(coords[i][1])) #TODO: move outside the for loop for all coords
            # print(x,y)

            crop = img[y - dist:y + dist, x - dist:x + dist]
            # plt.imshow(crop)
            # plt.show()
            count = crop.shape[0] * crop.shape[1]

            n_values.append(count)
            total_intensity.append(np.sum(crop))

    df = pd.DataFrame()
    df['total_intensity'] = total_intensity
    df['n_values'] = n_values
    df['mean_intensity'] = df['total_intensity'] / df['n_values']

    return df


def extract_bodypart_coords_from_DLC(h5, bodypart):
    dlc_df = pd.read_hdf(h5)
    bodypart_coordinates = dlc_df[dlc_df.columns.levels[0][0]][bodypart][['x', 'y']][:].values

    return bodypart_coordinates

#if __name__ == "__main__":
#
# # inputs
#
# projects= glob.glob("/Volumes/scratch/neurobiology/zimmer/active_sensing/zim06/zim2391/test_dlc_pipeline/20230515/data/worm*/")
# print(projects)
#
# for project_path in projects:
#     print(project_path)
#     red_path = glob.glob(os.path.join(project_path, '*Ch0/'))[0]
#     green_path = glob.glob(os.path.join(project_path, '*Ch1/'))[0]
#     print("red path is: ", red_path)
#     print("green path is: ", green_path)
#     h5 = glob.glob(os.path.join(red_path, "*mobnet*.h5"))[0]
#     print("h5 file is: ", h5)
#     red_image_path = os.path.join(red_path, "raw_stack_cropped.btf")
#     green_image_path = os.path.join(green_path, "raw_stack_cropped.btf")
#
#     # This should be loaded as dictionary with the bodypart as key and the coordinates as values
#     bodypart = "soma" #'soma'
#     dist = 2
#
#     coords = extract_bodypart_coords_from_DLC(h5, bodypart)
#
#     df = measure_signal_from_coords(red_image_path, coords, dist)
#     df.to_csv(os.path.join(red_path, "raw_stack_background_subtracted_masked_measurements.csv"))
#     df = measure_signal_from_coords(green_image_path, coords, dist)
#     df.to_csv(os.path.join(green_path, "raw_stack_background_subtracted_masked_measurements.csv"))
# print("done")