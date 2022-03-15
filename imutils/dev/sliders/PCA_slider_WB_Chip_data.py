import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr

#path to PC
path='/Users/ulises.rey/local_data/Oriana_worm3/w3_PCA.csv'
df=pd.read_csv(path, header=None)

#path to motor state annotation
path_motor_state='/Users/ulises.rey/local_data/Oriana_worm3/w3_state.csv'
motor_state_df=pd.read_csv(path_motor_state)

avg_win=10#167


x=df.iloc[:,[0]].rolling(window=avg_win, center=True).mean().values.flatten()
y=df.iloc[:,[1]].rolling(window=avg_win, center=True).mean().values.flatten()
z=df.iloc[:,[2]].rolling(window=avg_win, center=True).mean().values.flatten()
# x=df.iloc[:,[0]].rolling(window=avg_win, center=True).mean().values.flatten()
# y=df.iloc[:,[1]].rolling(window=avg_win, center=True).mean().values.flatten()
# z=df.iloc[:,[2]].rolling(window=avg_win, center=True).mean().values.flatten()
print(type(z))
print(len(z))
# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
#fig, ax = plt.subplots()
#plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')


ax1.scatter(x, y, z, lw=0.5, c=motor_state_df['state_color'], s=3)
ax1.plot(x,y,z, lw=0.5, c='grey', alpha=0.5)


plt.show()