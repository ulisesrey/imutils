import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import pandas as pd
import glob
import os

main_path='/Users/ulises.rey/local_data/BAG/traces/'
traces_files=glob.glob(os.path.join(main_path,'*traces.csv'))
for file in traces_files:
    df=pd.read_csv(file)
    df=df['ratiometric']
    fig,ax=plt.subplots(dpi=100)
    ax.plot(df)
    ax.set_title(os.path.basename(file))
plt.show()