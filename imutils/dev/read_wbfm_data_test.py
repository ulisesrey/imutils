import pandas as pd

import matplotlib.pyplot as plt

green = pd.read_hdf('/Volumes/scratch/neurobiology/zimmer/Charles/dlc_stacks/incomplete/C-NewBright6-2022_06_30/4-traces/green_traces.h5')

green.plot(subplots=True)
plt.show()