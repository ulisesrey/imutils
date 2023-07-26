#Based on PCA_Figure.py in the slider folder.

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

#path to PC
#wbfm
path="/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH/principal_components.csv"
    #'/Users/ulises.rey/local_data/Oriana_worm3/worm3_Oriana_PCA_5Components.csv'
#skeleton behaviour
#path='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/wbfm_worm3_PCA.csv'
df=pd.read_csv(path)#, header=None)

#path to motor state annotation
path_motor_state="/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm1/2022-11-27_15-14_ZIM2165_worm1_GC7b_Ch0-BH/principal_components.csv"
    #'/Users/ulises.rey/local_data/Oriana_worm3/w3_state.csv'
#skeleton behaviour
#path_motor_state='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/beh_annotation_16_subsamples_timeseries_with_long_reversals.csv'
motor_state_df=pd.read_csv(path_motor_state)

avg_win=10#167

#merge the dataframes
df['motor_state']=motor_state_df['PC3'] #motor_state_df['state']

#define writer
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Ulises Rey'), bitrate=1800)


#color dictionary
color_dict = {'forward': u'blue',
              'reversal': u'red',
              'sustained reversal': u'green',
              'ventral turn': u'orange',
              'dorsal turn': u'yellow'}

## all black colordict
# color_dict = {'forward': u'black',
#               'reversal': u'black',
#               'sustained reversal': u'black',
#               'ventral turn': u'black',
#               'dorsal turn': u'black'}

x=df.loc[:,['PC1']].rolling(window=avg_win, center=True).mean().values.flatten()
y=df.loc[:,['PC2']].rolling(window=avg_win, center=True).mean().values.flatten()
z=df.loc[:,['PC3']].rolling(window=avg_win, center=True).mean().values.flatten()

# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

line = ax1.scatter([], [], s=2)
scat=ax1.scatter([],[])
#ax1.set_axis_off()

initial_time = 0

# initialization function
def init():
    # creating an empty plot/frame
    #line.set_data([], [])
    line.set_offsets([])
    return line,

def animate(i):
    # t is a parameter
    t =  initial_time + i

    #line.set_data(xdata[initial_time:t], ydata[initial_time:t])
    #line.set_offsets(np.column_stack([xdata[initial_time:t], ydata[initial_time:t]]))
    line.set_offsets(np.column_stack([0, 0]))
    #scat.set_offsets(np.column_stack([xcentroid[initial_time:t], ycentroid[initial_time:t]]))#(, ycentroid[initial_time:t])
    ax1.scatter(x[t], y[t], z[t], lw=0.5, c=motor_state_df['state'].map(color_dict)[t], s=3)
    # pointer (Can't have it because it does not disappear, it stays
    #ax1.scatter(x[t], y[t], z[t], lw=0.5, c='black', s=10)
    #if t>251800: anim.pause()

    return  line, #scat #scat

anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(df), interval=1, blit=True)

ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

lim=0.2
ax1.set_xlim([-lim,lim])
ax1.set_ylim([-lim,lim])
ax1.set_zlim([-lim,lim])
anim.save('worm1_Ulises_PCA.mp4', writer=writer, dpi=300)
plt.show()