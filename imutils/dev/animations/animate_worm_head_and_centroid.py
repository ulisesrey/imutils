# function to animate worm track
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

#define writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

plt.style.use('seaborn-talk')

fig = plt.figure()
ax = plt.axes(xlim=[0, 6], ylim=[6, 11])
line, = ax.plot([], [], lw=2)
line2, = ax.plot([], [], lw=2)

#load data
path='/Users/ulises.rey/local_code/leopold_worms/data/concentration_change_2020-07-01_13-21-00_chemotaxisl_worm1.csv'
df=pd.read_csv(path)

#df=df.rolling(axis=0, window=16, win_type='gaussian', center=True, min_periods=1).mean(std=5)
df=df.rolling(axis=0, window=16, center=True, min_periods=1).median()
#pd.core.window.Rolling.median(df, axis=0, window=6, win_type='gaussian', center=True, min_periods=1)
# initialization function
def init():
    # creating an empty plot/frame
    line.set_data([], [])
    return line,


# lists to store x and y axis points
xdata= df['x_head_corrected']
ydata = df['y_head_corrected']

xcentroid = df['x_center']
ycentroid = df['y_center']

initial_time = 93100
# animation function
def animate(i):
    # t is a parameter
    t =  initial_time + i*100
    print(t)


    # x, y values to be plotted
    # x = t * np.sin(t)
    # y = t * np.cos(t)

    # appending new points to x, y axes points list
    # xdata.append(x)
    # ydata.append(y)
    print(xdata[t])
    print(ydata[t])

    line.set_data(xdata[initial_time:t], ydata[initial_time:t])
    line2.set_data(xcentroid[initial_time:t], ycentroid[initial_time:t])
    #if t>251800: anim.pause()

    return line, line2


# setting a title for the plot
plt.title('Creating a growing coil with matplotlib!')
# hiding the axis details
#plt.axis('off')

# call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=219, interval=10, blit=True)
plt.show()
anim.save('worm_tracks_head_and_centroid.avi', writer=writer)