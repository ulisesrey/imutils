# function to animate worm track
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

#define writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=5.2, metadata=dict(artist='Me'), bitrate=1800)

plt.style.use('seaborn-talk')

fig = plt.figure()
ax = plt.axes()
ax.set_xlim([-15, 5])
ax.set_ylim([-4, 4])
line, = ax.plot([], [], lw=2, c='black')

#load data
path='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/2021-03-04_16-17-30_worm3_ZIM2051-TablePosRecord.txt'
df=pd.read_csv(path)

# lists to store x and y axis points
xdata= df.loc[::16,'X']#df['x_head_corrected']
ydata = df.loc[::16,'Y']#df['y_head_corrected']

initial_time = 0

# initialization function
def init():
    # creating an empty plot/frame
    line.set_data([], [])
    return line,

# animation function
def animate(i):
    # t is a parameter
    t =  initial_time + i#*167
    print(t)
    # print(xdata[t])
    # print(ydata[t])


    line.set_data(xdata[initial_time:t], ydata[initial_time:t])
    return line,

# setting a title for the plot
#plt.title('Creating a growing coil with matplotlib!')
# hiding the axis details
plt.axis('off')

# call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=2125, interval=10, blit=True)
#plt.show()
anim.save('worm3_tracks_black_new2.mp4', writer=writer)