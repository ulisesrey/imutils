#It is not a slider actually

import matplotlib.pyplot as plt
import pandas as pd

#path to PC
#wbfm
path='/Users/ulises.rey/local_data/Oriana_worm3/worm3_Oriana_PCA_5Components.csv'
#skeleton behaviour
path='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/wbfm_worm3_PCA.csv'
df=pd.read_csv(path)#, header=None)

#path to motor state annotation
path_motor_state='/Users/ulises.rey/local_data/Oriana_worm3/w3_state.csv'
#skeleton behaviour
path_motor_state='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/beh_annotation_16_subsamples_timeseries_with_long_reversals.csv'
motor_state_df=pd.read_csv(path_motor_state)

avg_win=10#167

#merge the dataframes
df['motor_state']=motor_state_df['state']

#color dictionary
color_dict = {'forward': u'blue',
              'reversal': u'red',
              'sustained reversal': u'green',
              'ventral turn': u'orange',
              'dorsal turn': u'yellow'}

x=df.loc[:,['PC1']].rolling(window=avg_win, center=True).mean().values.flatten()
y=df.loc[:,['PC2']].rolling(window=avg_win, center=True).mean().values.flatten()
z=df.loc[:,['PC3']].rolling(window=avg_win, center=True).mean().values.flatten()

# Create the figure and the line that we will manipulate
fig = plt.figure(figsize=plt.figaspect(0.5), dpi=200)
plt.subplots_adjust(left=0.25, bottom=0.25)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')

ax1.scatter(x, y, z, lw=0.5, c=motor_state_df['state'].map(color_dict), s=3)
ax1.set_axis_off()

plt.show()