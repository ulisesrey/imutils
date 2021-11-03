import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr

# LOAD PCA DATA
path = '/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
# img_path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf.tif'
img_path = '/Volumes/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/btf_all_binary_after_unet_25493234/binary/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf'
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
plt.plot(cross_product_array[250:2500])
plt.hlines(0, xmin=250, xmax=2500)

# What to do with Nas?
# df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True)  # alternative change nans to zeros
features = np.arange(20, 70)  # Separating out the features (starting bodypart, ending bodypart)
data = df.loc[:, features].values
print('data shape: ', data.shape)

# PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data)
print(principalComponents.shape)
principalDf = pd.DataFrame(data=principalComponents,
                           columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5'])  # 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)

avg_win = 16
x = principalDf.loc[:, 'PC1'].rolling(window=avg_win).mean()
y = principalDf.loc[:, 'PC2'].rolling(window=avg_win).mean()
z = principalDf.loc[:, 'PC3'].rolling(window=avg_win).mean()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(x[250], y[250], z[250], c='green', s=150)
ax.scatter(x[250:2500], y[250:2500], z[250:2500], c=cross_product_array[250:2500], vmin=-4e-4, vmax=4e-4, cmap='bwr')
# ax.set_xlabel('PC1')
# ax.set_ylabel('PC2')
# ax.set_zlabel('PC3')
plt.show()
# plt.colorbar()

print('end')
