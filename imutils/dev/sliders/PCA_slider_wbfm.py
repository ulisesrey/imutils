import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr


# LOAD PCA DATA
#path to chemotaxis worm
path='/Users/ulises.rey/local_data/MondaySeminar3/2020-07-01_13-21-00_chemotaxisl_worm1-_spline_K.csv'#'/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'#'/Volumes/groups/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'
#path to wbfm worm3 worm
path ='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'#'/Volumes/groups/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'

#'/Volumes/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/btf_all_binary_after_new_unet_raw_eroded_twice_29322956_3_w_validation500steps_100epochs/binary_skeleton_output/2020-07-01_13-21-00_chemotaxisl_worm1-_spline_Y_coords.csv'#2020-07-01_18-36-25_control_worm6-_spline_K.csv'#'/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
#img_path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf.tif'
#path to chemotaxis worm
img_path='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/btf_all_binary_after_new_unet_raw_eroded_twice_29322956_3_w_validation500steps_100epochs/2020-07-01_13-21-00_chemotaxisl_worm1-channel-0-bigtiff.btf'
#path to wbfm worm3 worm
img_path='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf/2021-03-04_16-17-30_worm3_ZIM2051-channel-0-bigtiff_new.btf'
# has no reversals path='/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/all_good_skeleton/2020-06-30_18-17-47_chemotaxis_worm5_spline_K.csv'

#WBFM_worm
#path='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'#'/Volumes/groups/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-_spline_K.csv'
#'/Volumes/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/btf_all_binary_after_new_unet_raw_eroded_twice_29322956_3_w_validation500steps_100epochs/binary_skeleton_output/2020-07-01_13-21-00_chemotaxisl_worm1-_spline_Y_coords.csv'#2020-07-01_18-36-25_control_worm6-_spline_K.csv'#'/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'
#img_path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6-channel-0-bigtiff.btf.tif'
#img_path='/Volumes/project/neurobiology/zimmer/Ulises/wbfm/dat/btf_binary/2021-03-04_16-17-30_worm3_ZIM2051-channel-0-bigtiff_new.btf'

df=pd.read_csv(path, header=None)

df.shape

store=tiff.imread(img_path, aszarr=True)
img = zarr.open(store, mode='r')
print(img.shape)

#What to do with Nas?
#df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True) #alternative change nans to zeros
features = np.arange(5,90)# Separating out the features (starting bodypart, ending bodypart)
#time=np.arange(0,len(df))
data = df.loc[:, features].values
print('data shape: ', data.shape)

#PCA
# pca = PCA(n_components=5)
# principalComponents = pca.fit_transform(data)
# print(principalComponents.shape)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3','PC4','PC5'])# 'PC6', 'PC7', 'PC8','PC9','PC10'])
# print(principalDf.shape)

#PCA fit and transform separately
pca = PCA(n_components=5)
pca.fit(data)
principalComponents = pca.transform(df.loc[:, features].values)
print(principalComponents.shape)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3','PC4','PC5'])# 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)

avg_win=167#167
x=principalDf.loc[:,'PC1'].rolling(window=avg_win, center=True).mean()
y=principalDf.loc[:,'PC2'].rolling(window=avg_win, center=True).mean()
z=principalDf.loc[:,'PC3'].rolling(window=avg_win, center=True).mean()

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
#pointer makes the program crash, so it is now working now, see line 121
pointer, = ax1.plot(x.iloc[-1], y.iloc[-1], z.iloc[-1], 'go')

# lim_value=2#0.2
# ax1.set_xlim([-lim_value,lim_value])
# ax1.set_ylim([-lim_value,lim_value])
# ax1.set_zlim([-lim_value,lim_value])

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
    valmax=6000,
    valinit=1,
    valstep=1
)

cursor_ax = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
cursor_slider = Slider(
    ax=cursor_ax,
    label='current time',
    valmin=0,
    valmax=6000,
    valinit=5000,
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
    # the line below needed to be commented to avoid error
    print(x[current_time])
    print(y[current_time])
    print(z[current_time])
    # pointer.set_3d_properties(z[current_time])

    img_line.set_data(img[cursor_slider.val])
    fig.canvas.draw_idle()

# register the update function with each slider
start_slider.on_changed(update)
cursor_slider.on_changed(update)

plt.show()