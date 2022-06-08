#This is a GUI to visualize GCamp traces and Behaviour
## There is/was a test Jupytergui in centerline/dev/
# This is based on PCA_Slider_3D.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr
import os


# LOAD PCA DATA
#img_path='/Users/ulises.rey/local_data/BAG/2021-10-01_17-34-35_worm2_on-channel-0-behaviour-bigtiff.btf'
img_path='/Volumes/scratch/neurobiology/zimmer/Daniel/data/epiflourescence_calicum_imaging_chemotaxis_beads/op50/2022-01-25_13-18_recording_1_op50_6_beads_reached_ref/2022-01-25_13-18-06_recording_1_op50_6_beads-channel-0-behaviour-/2022-01-25_13-18-06_recording_1_op50_6_beads-channel-0-behaviour-bigtiff.btf'
    #'/Users/ulises.rey/local_data/Daniel_ZIM06_alignment/MondaySeminar3/untitled folder/2022-01-25_13-18-06_recording_1_op50_6_beads-channel-0-behaviour-bigtiff.btf'
    #GOOD ONE +'/Volumes/project/neurobiology/zimmer/Daniel_Mitic/data/raw_data/bag_zim06/circular/zim06Bag_1.5cm_circle/btiffs_behaviour/2021-10-01_18-11-05_worm3_on-channel-0-behaviour-bigtiff.btf'
signal_path='/Users/ulises.rey/local_data/Daniel_ZIM06_alignment/MondaySeminar3/untitled folder/2022-01-25_13-18_recording_1_trace.csv'
    #GOOD ONE='/Volumes/project/neurobiology/zimmer/Daniel_Mitic/data/zm06_bag/bag_circular_traces/1.5cm_circle/2021-10-01_18-11-03_worm3_on-channel-1-Andor9046bigtiff.btftraces.csv'
df=pd.read_csv(signal_path)

#df=pd.read_csv(path, header=None)

data=df['ratiometric'].values

df.shape

store=tiff.imread(img_path, aszarr=True)
img = zarr.open(store, mode='r')

#What to do with Nas?
#df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run


# Create the figure and the line that we will manipulate
fig = plt.figure()#(figsize=plt.figaspect(0.5), dpi=200)
#fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

img_line=ax2.imshow(img[0], origin='lower')#[200:-200,350:-350])#[100:-100,250:-250])#[200:800,200:700])

line_all, = ax1.plot(data, lw=1, color='k')
pointer, = ax1.plot(data[-1], 'go')
#pointer, = ax1.plot(x.iloc[-1], y.iloc[-1], z.iloc[-1], 'go')

ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')

# food part

# mirrored coordinates of the food lawn
#exp_folder='/Users/ulises.rey/local_data/Daniel_ZIM06_Alignment/recording_1_2021-12-30_04-00_op_50_gfp_5_beads_reached_ref/'
#stage_coords=pd.read_csv(os.path.join(exp_folder,'2021-12-30_04-00_recording1_gfp_bacteria-TablePosRecord.txt'))
stage_coords=pd.read_csv('/Users/ulises.rey/local_data/Daniel_ZIM06_alignment/MondaySeminar3/untitled folder/2022-01-25_13-18_recording_1_op50_6_beads-TablePosRecord.txt')
x=stage_coords['X'].values
y=stage_coords['Y'].values
#polygon coord
circle_coords_mm='/Users/ulises.rey/local_data/Daniel_ZIM06_alignment/MondaySeminar3/untitled folder/circle_coords_mm.csv'
opencv_solution=pd.read_csv(circle_coords_mm)


line_trajectory, = ax3.plot(stage_coords['X'], stage_coords['Y'], label='head position')
ax3.plot(opencv_solution['x'],opencv_solution['y'], label='food lawn')
ax3.axis('equal')
#ax3.plot(food_lawn, lw=2, color='y')

#distance to food: #needs to be written properly
# path_distance_to_food='/Volumes/scratch/neurobiology/zimmer/Daniel/beads/op50/good_recs/2022-01-25_13-18_recording_1_op50_6_beads_reached_ref/distance_worm_to_food.csv'
# distance_to_food=pd.read_csv(path_distance_to_food)
# line_distance_to_food, = ax1.plot(distance_to_food['0'], label='food distance')

# end food part

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'

cursor_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
cursor_slider = Slider(
    ax=cursor_ax,
    label='current time',
    valmin=4000,
    valmax=27000,
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
    valstep=1
)



# The function to be called anytime a slider's value changes
def update(val):
    window_size=int(window_slider.val)
    current_time=int(cursor_slider.val)
    print(current_time)
    #line_all.set_data(x[start_c:current_time])
    ax1.set_xlim([current_time-window_size/2, current_time+window_size/2])

    pointer.set_data(current_time,data[current_time])

    img_line.set_data(img[cursor_slider.val])#[200:-200,350:-350])

    # food part
    line_trajectory.set_data(x[0:current_time], y[0:current_time])
    fig.canvas.draw_idle()

# register the update function with each slider
window_slider.on_changed(update)
cursor_slider.on_changed(update)

plt.show()