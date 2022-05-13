if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import glob
    import os
    import numpy as np



    #variables
    image_center_coordinates = np.asarray((448, 464))
    px2mm_ratio = 0.00325
    # bodypart='head' future implementation
    project_path = '/Volumes/scratch/neurobiology/zimmer/ulises/active_sensing/epifluorescence_recordings/20220408/data/ZIM1661_worm3/'

    center_coords = pd.read_csv(glob.glob(os.path.join(project_path, '*TablePos*'))[0])
    center_coords = center_coords[['X', 'Y']].values
    # convert to negative so they fit the orientation of the image
    center_coords = - center_coords

    dlc_df = pd.read_hdf(glob.glob(os.path.join(project_path, '*behaviour*/*behaviour*.h5'))[0])
    points = dlc_df[dlc_df.columns.levels[0][0]]['head'][['x', 'y']][:].values

    # Specificy the point in the image that corresponds to stage position
    result = image_center_coordinates - np.asarray(points)

    result_mm = result * px2mm_ratio

    # somehow x coordinates needed to be substracted, whereas y coordinates added
    abs_coords_df=pd.DataFrame()
    abs_coords_df['X'] = center_coords[:, 0] - result_mm[:, 0]
    abs_coords_df['Y'] = center_coords[:, 1] + result_mm[:, 1]

    print(os.path.join(project_path, 'nose_coords_mm.csv'))
    abs_coords_df.to_csv(os.path.join(project_path, 'nose_coords_mm.csv'))

    print('end of generating head good coordinates')