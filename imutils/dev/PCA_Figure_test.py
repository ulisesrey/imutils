#this was a test to run this code locally using atom

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import mpld3



path='imutils/dev/2020-07-01_18-36-25_control_worm6_spline_K.csv'

# has no reversals path='/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/all_good_skeleton/2020-06-30_18-17-47_chemotaxis_worm5_spline_K.csv'
df=pd.read_csv(path, header=None)

df.shape
# What to do with Nas?
# df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True)  #alternative change nans to zeros
features = np.arange(30, 90)  #Separating out the features (starting bodypart, ending bodypart)
data = df.loc[:, features].values
print('data shape: ', data.shape)

# PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data)
print(principalComponents.shape)
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2', 'PC3','PC4','PC5'])  # 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)


# Second Figure
# define time
time=np.arange(0,principalDf.shape[0])
time=np.arange(193195,194500)  #easy
# time=np.arange(6600,7150) #challenging
# time=np.arange(6000,7500) #challenging

#np.arange(195003)#(142465)#(20000,30000)#np.arange(18000,42000)#np.arange(57500,62000)#np.arange(18000,42000)
#define figure
fig1, (ax0)=plt.subplots(figsize=(10,1000), dpi=50)
plt.rcParams.update({'font.size': 15})
for pc in principalDf:
    if pc == 'PC4': break
    ax=principalDf.loc[time,pc].plot(legend=True, figsize=(200,5), linewidth=2)
ax0.set_xlabel('Time (frames)', fontsize=15)
ax0.set_ylim([-.5,.5])
ax0.set_ylabel('',fontsize=15)
#plt.show()


mpl.rcParams['legend.fontsize'] = 10

fig2=plt.figure(figsize=(10,10), dpi=50)

ax1 = fig2.gca(projection='3d')
#time=np.arange(193195,196000)
avg_win=167
x=principalDf.loc[time,'PC1'].rolling(window=avg_win).mean()
y=principalDf.loc[time,'PC2'].rolling(window=avg_win).mean()
z=principalDf.loc[time,'PC3'].rolling(window=avg_win).mean()

#ax1.plot(x, y, z, label='BH PC')
#ax1.scatter(x, y, z, label='BH PC')
ax1.scatter(x, y, z, c=time, label='BH PC')
ax1.legend()

plt.show()

plt.rcParams.update({'font.size': 10})


#3rd figure3

# avg_win=16
# x=principalDf.loc[:,'PC1'].rolling(window=avg_win).mean()
# y=principalDf.loc[:,'PC2'].rolling(window=avg_win).mean()
# fig3=plt.figure(dpi=200)
# plt.scatter(x,y, c=np.arange(len(x)), s=1, cmap='viridis')
# plt.scatter(x.to_numpy()[avg_win+1],y.to_numpy()[avg_win+1], c='g', edgecolors='k')
# plt.scatter(x.to_numpy()[-1],y.to_numpy()[-1], c='r', edgecolors='k')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# #lim=0.05#0.25
# # plt.xlim([-lim, lim])
# # plt.ylim([-lim, lim])
# plt.colorbar()
# plt.show()
