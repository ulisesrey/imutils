import tifffile as tiff
from skimage.morphology import skeletonize
import numpy as np

with tiff.TiffFile("/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20221127/data/ZIM2165_Gcamp7b_worm2/2022-11-27_15-34_ZIM2165_worm2_GC7b_Ch0-BH/2022-11-27_15-34_ZIM2165_worm2_GC7b_Ch0-BHbigtiff_AVG_background_subtracted_normalised_unet_segmented_weights_5358068_1_mask_coil_segmented_mask.btf") as tif:
    for page in tif.pages:
        img= page.asarray()

        img[img>1]=1
        img[img<1]=0
        skeleton = skeletonize(img)