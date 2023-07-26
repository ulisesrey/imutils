import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns

heatmap_path = "/Users/ulises.rey/local_data/Oriana_worm3/simple_deltaFOverF_bc_50.csv"

df = pd.read_csv(heatmap_path, header=None)

df.head()

sns.heatmap(df)

plt.show()