# check this to make it animated!
# https://stackoverflow.com/questions/22254776/python-matplotlib-set-array-takes-exactly-2-arguments-3-given

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr
import glob
import os
from curvature.src.annotate_reversals import *

# LOAD PCA DATA
#path = '/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
#NEW WBFM worms


main_paths = glob.glob("/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221130/data/*worm*/*Ch0-BH") #[0]

for main_path in main_paths:
    print(main_path)
    path = glob.glob(os.path.join(main_path,"skeleton_spline_K_signed_avg.csv"))[0]

    avg_win = 1

    df = pd.read_csv(path, header=None)

    # What to do with Nas?
    # df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
    df.fillna(0, inplace=True)  # alternative change nans to zeros
    features = np.arange(30, 80)  # Separating out the features (starting bodypart, ending bodypart)
    data = df.loc[:, features].values


    #Calculate PCA
    # pca = PCA(n_components=3)
    # principalComponents = pca.fit_transform(data)
    # print(principalComponents.shape)
    # principal_components_df = pd.DataFrame(data=principalComponents,
    #                            columns=['PC1', 'PC2', 'PC3'])#, 'PC4', 'PC5'])  # 'PC6', 'PC7', 'PC8','PC9','PC10'])
    # print(principal_components_df.shape)



    principal_components_df = pd.read_csv(glob.glob(os.path.join(main_path,"principal_components.csv"))[0])

    #calculate cross product here
    # pc1_pc2_df = extract_vectors_from_PC_df(principal_components_df, avg_win=avg_win)
    # cross_product_df = calculate_cross_product(pc1_pc2_df)
    # cross_product_array = np.array(cross_product_df['Cross_Product'].rolling(window=avg_win, center=True).mean())
    # cross_product_array=cross_product_array*-1

    # calculate total curvature here
    signed_curvature = data.sum(axis=1)


    x = principal_components_df.loc[:, 'PC1'].rolling(window=avg_win).mean()
    y = principal_components_df.loc[:, 'PC2'].rolling(window=avg_win).mean()
    z = principal_components_df.loc[:, 'PC3'].rolling(window=avg_win).mean()
    color_variable = signed_curvature

    start_frame, end_frame = 0, -1#250, 100000 #2050, 4000#250, 100000

    def plot_3d_pca(x,y,z,color_variable, start_frame,end_frame):
        fig = plt.figure(dpi=200)
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x[start_frame:end_frame], y[start_frame:end_frame], z[start_frame:end_frame],
                   c=color_variable[start_frame:end_frame], s=.25, vmin=-1, vmax=1, cmap='PRGn')
        ax.set_xlabel('PC1', fontsize=15, labelpad=10)
        ax.set_ylabel('PC2', fontsize=15, labelpad=10)
        ax.set_zlabel('PC3', fontsize=15, labelpad=10)
        ax.tick_params(labelsize=7)
        #ax.set_axis_off()

        plt.show()

    plot_3d_pca(x,y,z,color_variable, start_frame,end_frame)


    plt.scatter(z,signed_curvature)
    plt.show()

print("end")