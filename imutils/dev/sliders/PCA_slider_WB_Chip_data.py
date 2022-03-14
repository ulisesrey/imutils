import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA
import tifffile as tiff
import zarr

path='/Users/ulises.rey/local_data/Oriana_worm3/w3_PCA.csv'
df=pd.read_csv(path, header=None)
avg_win=167#167
x=df.iloc[:,[10]].values.flatten()
y=df.iloc[:,[20]].values.flatten()
z=df.iloc[:,[30]].values.flatten()
# x=df.iloc[:,[0]].rolling(window=avg_win, center=True).mean().values.flatten()
# y=df.iloc[:,[1]].rolling(window=avg_win, center=True).mean().values.flatten()
# z=df.iloc[:,[2]].rolling(window=avg_win, center=True).mean().values.flatten()
print(type(z))
# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
#fig, ax = plt.subplots()
#plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')


ax1.plot(x, y, z, lw=0.5, color='grey')
ax1.plot(x,y,z, lw=2)


plt.show()