#This is a GUI to visualize GCamp traces and Behaviour
## There is/was a test Jupytergui in centerline/dev/
# This is based on PCA_Slider_3D.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff


# LOAD PCA DATA
img_path='/Users/ulises.rey/local_data/BAG/2021-10-01_17-34-35_worm2_on-channel-0-behaviour-bigtiff.btf'
signal_path='/Users/ulises.rey/local_data/BAG/traces/2021-10-01_17-34-34_worm2_on-channel-1-Andor9046bigtiff.btftraces.csv'
df=pd.read_csv(signal_path)

#df=pd.read_csv(path, header=None)

data=df['ratiometric'].values

df.shape

img=tiff.imread(img_path)
#What to do with Nas?
#df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run


# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
#fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


img_line=ax2.imshow(img[0])

line_all, = ax1.plot(data, lw=1, color='k')
pointer, = ax1.plot(data[-1], 'go')
#pointer, = ax1.plot(x.iloc[-1], y.iloc[-1], z.iloc[-1], 'go')

ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'

cursor_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
cursor_slider = Slider(
    ax=cursor_ax,
    label='current time',
    valmin=0,
    valmax=30000,
    valinit=100,
    valstep=1
)

window_ax = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
window_slider = Slider(
    ax=window_ax,
    label='Window size',
    valmin=2,
    valmax=30000,
    valinit=200,
    valstep=2
)



# The function to be called anytime a slider's value changes
def update(val):
    window_size=int(window_slider.val)
    current_time=int(cursor_slider.val)
    print(current_time)
    #line_all.set_data(x[start_c:current_time])
    ax1.set_xlim([current_time-window_size/2, current_time+window_size/2])

    pointer.set_data(current_time,data[current_time])


    img_line.set_data(img[cursor_slider.val])
    fig.canvas.draw_idle()

# register the update function with each slider
window_slider.on_changed(update)
cursor_slider.on_changed(update)

plt.show()