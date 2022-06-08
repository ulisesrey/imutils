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
path = '/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'#'/Volumes/groups/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'
#'/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
# img_path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf.tif'
img_path='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-channel-0-bigtiff_new.btf'
# has no reversals path='/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/all_good_skeleton/2020-06-30_18-17-47_chemotaxis_worm5_spline_K.csv'

df = pd.read_csv(path, header=None)

df.shape

store = tiff.imread(img_path, aszarr=True)
img = zarr.open(store, mode='r')
print(img.shape)

# read cross product calculated elsewhere:
cross_product_path = '/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff_cross_product.csv'
cross_product_df = pd.read_csv(cross_product_path)
cross_product_array = np.array(cross_product_df['Cross Product'].rolling(window=16, center=True).mean())
# plt.plot(cross_product_array[250:2500])
# plt.hlines(0, xmin=250, xmax=2500)

# What to do with Nas?
# df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True)  # alternative change nans to zeros
features = np.arange(5, 90)  # Separating out the features (starting bodypart, ending bodypart)
data = df.loc[:, features].values
print('data shape: ', data.shape)

# PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data)
print(principalComponents.shape)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])  # 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)

avg_win = 167*1
x = principalDf.loc[:, 'PC1'].rolling(window=avg_win).mean()
y = principalDf.loc[:, 'PC2'].rolling(window=avg_win).mean()
z = principalDf.loc[:, 'PC3'].rolling(window=avg_win).mean()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#green dot at first frame
#ax.scatter(x[250], y[250], z[250], c='green', s=150)
start_frame, end_frame=250,50000
ax.scatter(x[start_frame:end_frame], y[start_frame:end_frame], z[start_frame:end_frame], c=cross_product_array[start_frame:end_frame], s=.25, vmin=-4e-4, vmax=4e-4, cmap='bwr')
#ax.scatter(x, y, z, c=cross_product_array, s=.25, vmin=-4e-4, vmax=4e-4, cmap='bwr')


ax.set_xlabel('PC1', fontsize = 12)
ax.set_ylabel('PC2', fontsize = 12)
ax.set_zlabel('PC3', fontsize = 12)
#remove ticks?
ax.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)


ax.set_axis_off()
plt.show()
# plt.colorbar()

print('end')
