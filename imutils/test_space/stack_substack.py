from imutils.src.imfunctions import stack_subsample
import numpy as np

input_filepath = '/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/wbfm_2nd_unet_training_coiled_shape/2022-01-27_21-26-53_worm4-channel-0-behaviour-bigtiff_AVG_background_substracted_unet_segmented_2626261_1.btf'
output_filepath = '/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/wbfm_2nd_unet_training_coiled_shape/2022-01-27_21-26-53_worm4-channel-0-behaviour-bigtiff_AVG_background_substracted_unet_segmented_2626261_1_subsample.btf'
range = np.arange(0,50000,100)

stack_subsample(input_filepath, output_filepath, range)