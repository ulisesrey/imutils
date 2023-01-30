# First part of the section 1. of the napari tracking tutorial
# https://napari.org/stable/tutorials/tracking/cell_tracking.html

import os
import napari

import numpy as np
import pandas as pd

from skimage.io import imread
from skimage.measure import regionprops_table

PATH = '/Users/ulises.rey/local_data/Fluo-N3DH-CE'
NUM_IMAGES = 195

def load_image(idx: int):
    """Load an image from the sequence.

    Parameters
    ----------
    idx : int
        Index of the image to load.

    Returns
    -------
    image : np.ndarray
       The image specified by the index, idx
    """
    filename = os.path.join(PATH, '01_GT/TRA', f'man_track{idx:0>3}.tif')
    return imread(filename)

def regionprops_plus_time(idx):
    """Return the unique track label, centroid and time for each track vertex.

    Parameters
    ----------
    idx : int
        Index of the image to calculate the centroids and track labels.

    Returns
    -------
    data_df : pd.DataFrame
       The dataframe of track data for one time step (specified by idx).
    """
    props = regionprops_table(stack[idx, ...], properties=('label', 'centroid'))
    props['frame'] = np.full(props['label'].shape, idx)
    return pd.DataFrame(props)


stack = np.asarray([load_image(i) for i in range(NUM_IMAGES)])

data_df_raw = pd.concat([regionprops_plus_time(idx) for idx in range(NUM_IMAGES)]).reset_index(drop=True)

# sort the data lexicographically by track_id and time
data_df = data_df_raw.sort_values(['label', 'frame'], ignore_index=True)

# create the final data array: track_id, T, Z, Y, X
data = data_df.loc[:, ['label', 'frame', 'centroid-0', 'centroid-1', 'centroid-2']].to_numpy()
print(data.head(10))

napari.view_tracks(data, name='tracklets')
napari.run()