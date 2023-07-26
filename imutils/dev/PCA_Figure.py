#It is not a slider actually

import matplotlib.pyplot as plt
import pandas as pd

#path to PC
#wbfm
path='/Users/ulises.rey/local_data/Oriana_worm3/worm3_Oriana_PCA_5Components.csv'
#skeleton behaviour
#path='/Users/ulises.rey/local_data/worm3_wbfm_PCA/wbfm_worm3_PCA.csv'
# '/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/wbfm_worm3_PCA.csv'
df=pd.read_csv(path)#, header=None)

#path to motor state annotation
path_motor_state='/Users/ulises.rey/local_data/Oriana_worm3/w3_state.csv'
#skeleton behaviour
#path_motor_state='/Users/ulises.rey/local_data/worm3_wbfm_PCA/beh_annotation_16_subsamples_timeseries_with_long_reversals.csv'
# #'/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/beh_annotation_16_subsamples_timeseries_with_long_reversals.csv'
motor_state_df=pd.read_csv(path_motor_state)

avg_win=10#167

#merge the dataframes
df['motor_state']=motor_state_df['state']

#subsample

#color dictionary
color_dict = {'forward': u'purple',
              'reversal': u'red',
              'sustained reversal': u'red',
              'ventral turn': u'purple',
              'dorsal turn': u'purple'}
## all black colordict
# color_dict = {'forward': u'black',
#               'reversal': u'black',
#               'sustained reversal': u'black',
#               'ventral turn': u'black',
#               'dorsal turn': u'black'}
print(len(df))
#oriana chip
start_time=1000
end_time=1590
# oriana chip new
# start_time=0
# end_time=2941
#behaviour
# start_time=1400
# end_time=2200
# start_time=1400
# end_time=1800


x=df.loc[start_time:end_time,['PC1']].rolling(window=avg_win, center=True).mean().values.flatten()
y=df.loc[start_time:end_time,['PC2']].rolling(window=avg_win, center=True).mean().values.flatten()
z=df.loc[start_time:end_time,['PC3']].rolling(window=avg_win, center=True).mean().values.flatten()

# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

#ax1.plot(x, y, z, lw=0.5, c='black', alpha=.6)
ax1.scatter(x, y, z, lw=0.5, c=motor_state_df.loc[start_time:end_time,'state'].map(color_dict), s=5)
#ax1.set_axis_off()


ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.set_zlabel('PC3')

lim=0.4
ax1.set_xlim([-lim,lim])
ax1.set_ylim([-lim,lim])
ax1.set_zlim([-lim,lim])
#plt.show()

#plt.savefig('PCA_freely_moving_empty.png', format='png', dpi=200)

print()

# Plot 2D

#fig2 = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
fig2, ax2 = plt.subplots()
#ax2 = fig.add_subplot(1, 1, 1)
ax2.scatter(x, y, lw=0.5, c=motor_state_df.loc[start_time:end_time,'state'].map(color_dict), s=20)
plt.show()