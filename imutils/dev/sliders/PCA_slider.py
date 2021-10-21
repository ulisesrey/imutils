import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
from sklearn.decomposition import PCA


# LOAD PCA DATA
path='/Users/ulises.rey/local_code/PCA_test/2020-07-01_18-36-25_control_worm6_spline_K.csv'

# has no reversals path='/groups/zimmer/Ulises/wbfm/chemotaxis_assay/2020_Only_behaviour/all_good_skeleton/2020-06-30_18-17-47_chemotaxis_worm5_spline_K.csv'
df=pd.read_csv(path, header=None)

df.shape
#What to do with Nas?
#df.dropna(inplace=True) #Drop NaNs, required otherwise pca.fit_transform(x) does not run
df.fillna(0, inplace=True) #alternative change nans to zeros
features = np.arange(30,90)# Separating out the features (starting bodypart, ending bodypart)
data = df.loc[:, features].values
print('data shape: ', data.shape)

#PCA
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data)
print(principalComponents.shape)
principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3','PC4','PC5'])# 'PC6', 'PC7', 'PC8','PC9','PC10'])
print(principalDf.shape)

avg_win=167
x=principalDf.loc[:,'PC1'].rolling(window=avg_win).mean()
y=principalDf.loc[:,'PC2'].rolling(window=avg_win).mean()
z=principalDf.loc[:,'PC3'].rolling(window=avg_win).mean()


# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)

line, = plt.plot(x, lw=2)
ax.set_xlabel('Time [s]')

# Make a horizontal slider to control the frequency.
axcolor = 'lightgoldenrodyellow'
start_ax = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
start_slider = Slider(
    ax=start_ax,
    label='Starting time',
    valmin=0,
    valmax=15000,
    valinit=0,
    valstep=1
)

end_ax = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
end_slider = Slider(
    ax=end_ax,
    label='ending time',
    valmin=0,
    valmax=50000,
    valinit=1000,
    valstep=1
)


# The function to be called anytime a slider's value changes
def update(val):
    start_c=int(start_slider.val)
    end_c=int(end_slider.val)
    #line.set_data(np.linspace(start_c, end_c, end_c-start_c),t[start_c:end_c])
    line.set_xdata(np.linspace(start_c,end_c, end_c-start_c))
    line.set_ydata(x[start_c:end_c])
    ax.set_xlim([0,end_c])
    fig.canvas.draw_idle()

# register the update function with each slider
start_slider.on_changed(update)
end_slider.on_changed(update)

plt.show()