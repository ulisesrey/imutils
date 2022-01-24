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
img_path='/Volumes/project/neurobiology/zimmer/Daniel_Mitic/data/raw_data/bag_zim06/circular/zim06Bag_1.5cm_circle/btiffs_behaviour/2021-10-01_18-11-05_worm3_on-channel-0-behaviour-bigtiff.btf'
signal_path='/Volumes/project/neurobiology/zimmer/Daniel_Mitic/data/zm06_bag/bag_circular_traces/1.5cm_circle/2021-10-01_18-11-03_worm3_on-channel-1-Andor9046bigtiff.btftraces.csv'
df=pd.read_csv(signal_path)

#df=pd.read_csv(path, header=None)

data=df['ratiometric'].values

df.shape

store=tiff.imread(img_path, aszarr=True)
img = zarr.open(store, mode='r')
#What to do with Nas?
#df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run


# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
#fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

img_line=ax2.imshow(img[0][200:-200,350:-350])#[100:-100,250:-250])#[200:800,200:700])

line_all, = ax1.plot(data, lw=1, color='k')
pointer, = ax1.plot(data[-1], 'go')
#pointer, = ax1.plot(x.iloc[-1], y.iloc[-1], z.iloc[-1], 'go')

ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')

# food part

# mirrored coordinates of the food lawn
exp_folder='/Users/ulises.rey/local_data/Daniel_ZIM06_Alignment/recording_1_2021-12-30_04-00_op_50_gfp_5_beads_reached_ref/'
stage_coords=pd.read_csv(os.path.join(exp_folder,'2021-12-30_04-00_recording1_gfp_bacteria-TablePosRecord.txt'))
x=stage_coords['X'].values
y=stage_coords['Y'].values
#polygon coord
polygon_h_mirr=np.array([[1343,640],
[1425.5,695.5],
[1521.5,755.5],
[1594.5,815.5],
[1640.5,873.5],
[1670.5,920.5],
[1695.5,981.5],
[1706.5,1010.5],
[1731.5,1033.5],
[1717.5,1067.5],
[1702.5,1111.5],
[1678.5,1173.5],
[1646.5,1240.5],
[1615.5,1293.5],
[1582.5,1327.5],
[1556.5,1372.5],
[1515.5,1370.5],
[1480.5,1373.5],
[1444.5,1394.5],
[1420.5,1428.5],
[1367.5,1428.5],
[1339.5,1435.5],
[1310.5,1455.5],
[1265.5,1480.5],
[1182.5,1483.5],
[1104.5,1491.5],
[1035.5,1491.5],
[1009.5,1474.5],
[990.5,1445.5],
[969.5,1438.5],
[948.5,1448.5],
[923.5,1469.5],
[910.5,1478.5],
[901.5,1473.5],
[896.5,1458.5],
[898.5,1448.5],
[905.5,1436.5],
[917.5,1413.5],
[923.5,1401.5],
[920.5,1387.5],
[899.5,1377.5],
[875.5,1373.5],
[865.5,1362.5],
[858.5,1332.5],
[832.5,1307.5],
[800.5,1295.5],
[792.5,1274.5],
[804.5,1247.5],
[816.5,1221.5],
[821.5,1192.5],
[802.5,1168.5],
[790.5,1164.5],
[778.5,1158.5],
[792.5,1139.5],
[808.5,1123.5],
[812.5,1113.5],
[796.5,1094.5],
[769.5,1078.5],
[759.5,1054.5],
[750.5,1035.5],
[752.5,1012.5],
[755.5,985.5],
[757.5,958.5],
[771.5,920.5],
[775.5,893.5],
[787.5,875.5],
[800.5,872.5],
[821.5,859.5],
[829.5,839.5],
[835.5,808.5],
[902.5,737.5],
[951.5,685.5],
[1001.5,645.5],
[1051.5,611.5],
[1117.5,592.5],
[1172.5,590.5],
[1228.5,593.5],
[1274.5,603.5],
[1311.5,627.5]])

line_trajectory, = ax3.plot(stage_coords['X'], stage_coords['Y'], label='head position')
ax3.axis('equal')
#ax3.plot(food_lawn, lw=2, color='y')

# end food part

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'

cursor_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
cursor_slider = Slider(
    ax=cursor_ax,
    label='current time',
    valmin=4000,
    valmax=6000,
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

    img_line.set_data(img[cursor_slider.val][200:-200,350:-350])

    # food part
    line_trajectory.set_data(x[0:current_time], y[0:current_time])
    fig.canvas.draw_idle()

# register the update function with each slider
window_slider.on_changed(update)
cursor_slider.on_changed(update)

plt.show()