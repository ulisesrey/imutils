import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from pathlib import Path
import os
import tifffile as tiff

#script for Tanja to plot max projections of each condition
main_path='/Volumes/scratch/neurobiology/zimmer/Tanja/project_food_aversion/quantified/max_projections'

genotypes = ['ZIM2156']

concentrations = ['op50']

images=glob.glob(os.path.join(main_path,'*.tif'))

for genotype in genotypes:
    print(genotype)
    for concentration in concentrations:
        print(concentration)
        for image in images:
            if genotype in image and concentration in image:
                img=tiff.imread(image)
                print(image)
                plt.imshow(img)
                plt.show()

