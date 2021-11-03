import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr


# LOAD PCA DATA
path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
#img_path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf.tif'
img_path='/Volumes/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/btf_all_binary_after_unet_25493234/binary/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf'
# has no reversals path='/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/all_good_skeleton/2020-06-30_18-17-47_chemotaxis_worm5_spline_K.csv'

df=pd.read_csv(path, header=None)

df.shape

store=tiff.imread(img_path, aszarr=True)
img = zarr.open(store, mode='r')
print(img.shape)
#What to do with Nas?
#df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True) #alternative change nans to zeros
features = np.arange(0,99)# Separating out the features (starting bodypart, ending bodypart)
data = df.loc[:, features].values
print('data shape: ', data.shape)

#PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data)
print(principalComponents.shape)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3','PC4','PC5'])# 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)

avg_win=167
x=principalDf.loc[:,'PC1'].rolling(window=avg_win).mean()
y=principalDf.loc[:,'PC2'].rolling(window=avg_win).mean()
z=principalDf.loc[:,'PC3'].rolling(window=avg_win).mean()


# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
#fig, ax = plt.subplots()
#plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)

line_all, = ax1.plot(x,y,z, lw=0.5, color='grey')
line, = ax1.plot(x,y,z, lw=2)
print(type(x))
print(x.iloc[-1])
pointer, = ax1.plot(x.iloc[-1], y.iloc[-1], z.iloc[-1], 'go')
lim_value=0.2
ax1.set_xlim([-lim_value,lim_value])
ax1.set_ylim([-lim_value,lim_value])
ax1.set_zlim([-lim_value,lim_value])

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

img_line=ax2.imshow(img[0])

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'
start_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
start_slider = Slider(
    ax=start_ax,
    label='Starting time',
    valmin=0,
    valmax=200001,
    valinit=0,
    valstep=1
)

cursor_ax = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
cursor_slider = Slider(
    ax=cursor_ax,
    label='current time',
    valmin=0,
    valmax=200001,
    valinit=100,
    valstep=1
)

# The function to be called anytime a slider's value changes
def update(val):
    start_c=int(start_slider.val)#*167
    current_time=int(cursor_slider.val)#*167

    line_all.set_xdata(x[start_c:current_time])
    line_all.set_ydata(y[start_c:current_time])
    line_all.set_3d_properties(z[start_c:current_time])

    line.set_xdata(x[current_time-600:current_time])
    line.set_ydata(y[current_time-600:current_time])
    line.set_3d_properties(z[current_time-600:current_time])

    pointer.set_xdata(x[current_time])
    pointer.set_ydata(y[current_time])
    pointer.set_3d_properties(z[current_time])

    img_line.set_data(img[cursor_slider.val])
    fig.canvas.draw_idle()

# register the update function with each slider
start_slider.on_changed(update)
cursor_slider.on_changed(update)

plt.show()