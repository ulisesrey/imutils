# function to animate worm track
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

#load data
path='/Users/ulises.rey/local_code/leopold_worms/data/concentration_change_2020-07-01_13-21-00_chemotaxisl_worm1.csv'
df=pd.read_csv(path)

y_fit_mat=np.load('y_fit_matrix.npy')

#define writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

plt.style.use('seaborn-talk')

fig = plt.figure()
ax = plt.axes(xlim=[0, 6], ylim=[6.5, 10.5])
#line, = ax.plot([], [], lw=2)
line = ax.scatter([], [], s=2)
scat = ax.scatter([], [], s=2, c='k')

#plot gradient
ax.imshow(y_fit_mat.T, extent=[0,6, 6, 11], cmap='YlOrBr', alpha=0.5)



#df=df.rolling(axis=0, window=16, win_type='gaussian', center=True, min_periods=1).mean(std=5)
#df=df.rolling(axis=0, window=32, center=True, min_periods=1).median()

# initialization function
def init():
    # creating an empty plot/frame
    #line.set_data([], [])
    line.set_offsets([])
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

    print(xdata[t])
    print(ydata[t])

    #line.set_data(xdata[initial_time:t], ydata[initial_time:t])
    #line.set_offsets(np.column_stack([xdata[initial_time:t], ydata[initial_time:t]]))
    line.set_offsets(np.column_stack([0, 0]))
    scat.set_offsets(np.column_stack([xcentroid[initial_time:t], ycentroid[initial_time:t]]))#(, ycentroid[initial_time:t])
    #if t>251800: anim.pause()

    return  line, scat #scat


# setting a title for the plot
#plt.title('Creating a growing coil with matplotlib!')
# hiding the axis details
plt.axis('off')

# call the animator
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=219, interval=10, blit=True)
plt.show()
#anim.save('worm_tracks_centroid.mp4', writer=writer, dpi=200)