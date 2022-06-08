# check this to make it animated!
# https://stackoverflow.com/questions/22254776/python-matplotlib-set-array-takes-exactly-2-arguments-3-given

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr

# LOAD PCA DATA
#path = '/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
path='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'#'/Volumes/groups/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'


df = pd.read_csv(path, header=None)

df.shape


avg_win = 167*1

# read cross product calculated elsewhere:
#cross_product_path = '/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff_cross_product.csv'
cross_product_path = '/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K_cross_product.csv'
cross_product_df = pd.read_csv(cross_product_path)
cross_product_array = np.array(cross_product_df['Cross Product'].rolling(window=avg_win, center=True).mean())
cross_product_array=cross_product_array*-1
# plt.plot(cross_product_array[250:2500])
# plt.hlines(0, xmin=250, xmax=2500)

# What to do with Nas?
# df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True)  # alternative change nans to zeros
features = np.arange(3, 90)  # Separating out the features (starting bodypart, ending bodypart)
data = df.loc[:, features].values
print('data shape: ', data.shape)

# PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(data)
print(principalComponents.shape)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['PC1', 'PC2', 'PC3'])#, 'PC4', 'PC5'])  # 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)

x = principalDf.loc[:, 'PC1'].rolling(window=avg_win).mean()
y = principalDf.loc[:, 'PC2'].rolling(window=avg_win).mean()
z = principalDf.loc[:, 'PC3'].rolling(window=avg_win).mean()


start_frame, end_frame = 250, 100000 #2050, 4000#250, 100000

def plot_3d_pca(x,y,z,cross_product_array, start_frame,end_frame):
    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection='3d')

    ax.scatter(x[start_frame:end_frame], y[start_frame:end_frame], z[start_frame:end_frame],
               c=cross_product_array[start_frame:end_frame], s=.25, vmin=-1e-4, vmax=1e-4, cmap='bwr')
    ax.set_xlabel('PC1', fontsize=15, labelpad=10)
    ax.set_ylabel('PC2', fontsize=15, labelpad=10)
    ax.set_zlabel('PC3', fontsize=15, labelpad=10)
    ax.tick_params(labelsize=7)
    ax.set_axis_off()

    plt.show()

plot_3d_pca(x,y,z,cross_product_array, start_frame,end_frame)