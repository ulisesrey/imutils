#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
import tifffile as tiff
from tifffile import imsave
from tifffile import TiffWriter
from PIL import Image
import natsort
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--i_path", required=True, help="path to input images")
ap.add_argument("-o", "--o_path", required=True, help="path to the output image")
ap.add_argument("-n", "--name", required=True, help="name of the output file")
ap.add_argument("-c", "--compression", help="Compression, set to 0 for none", default=0)
args = vars(ap.parse_args())


i_directory=(args["i_path"])
o_directory=(args["o_path"])
os.makedirs(o_directory)
filename=os.listdir(i_directory)[0]
print(filename)
IM=tiff.imread(os.path.join(i_directory,filename))
#fold increase for better visuTiffWriteron of the very 'short' z axis. Set to 1 for no increase.
#set it to have the same pixel size as XY (0.3um vs 1um zstep size)
fold_increase=3

#creates an empty matrix for the lower right part of the image. It should have the dimensions of Z-stacks times the fold increase. (now 150)
corner_matrix=np.full((fold_increase*IM.shape[0],fold_increase*IM.shape[0]),100, dtype='uint16')

# for filename in os.listdir(i_directory):
# for n in np.arange(0,np.shape((os.listdir(i_directory)))[0]):
# for n in np.arange(0,60):
with TiffWriter((o_directory + args["name"]+'.tif'), bigtiff=True) as tif_w:
    for filename in natsort.natsorted(os.listdir(i_directory)):
        print(filename)
        if filename.endswith(".stk"):
            IM = tiff.imread(os.path.join(i_directory, filename))
            # creates max projections in all axes
            IM_MAX_0 = np.max(IM, axis=0)
            IM_MAX_1 = np.max(IM, axis=1)
            IM_MAX_2 = np.max(IM, axis=2)
            # rotates IM_MAX_2
            rot_IM_MAX_2=np.transpose(IM_MAX_2)

            # extends the YZ and XZ max projection for better visualization based on input fold_increase variable
            rep_rot_IM_MAX_2 = np.repeat(rot_IM_MAX_2, fold_increase, axis=1)
            rep_IM_MAX_1 = np.repeat(IM_MAX_1, fold_increase, axis=0)

            # normalizes image intensities (not working fine somehow)
            # IM_MAX_0=cv2.normalize(IM_MAX_0, IM_MAX_0, alpha=np.amin(IM_MAX_0), beta=np.amax(IM_MAX_0), norm_type=cv2.NORM_MINMAX)
            # rep_rot_IM_MAX_2=cv2.normalize(rep_rot_IM_MAX_2, rep_rot_IM_MAX_2, alpha=np.amin(rep_rot_IM_MAX_2), beta=np.amax(rep_rot_IM_MAX_2), norm_type=cv2.NORM_MINMAX)
            # rep_IM_MAX_1=cv2.normalize(rep_IM_MAX_1, rep_IM_MAX_1, alpha=np.amin(rep_IM_MAX_1), beta=np.amax(rep_IM_MAX_1), norm_type=cv2.NORM_MINMAX)

            # concatenate the different max projections into one image
            vert_conc_1 = np.concatenate((IM_MAX_0, rep_IM_MAX_1), axis=0)
            vert_conc2 = np.concatenate((rep_rot_IM_MAX_2, corner_matrix), axis=0)
            final_image = np.concatenate((vert_conc_1, vert_conc2), axis=1)

            # convert to 16bit
            #final_image16bit = np.int16(final_image)

            # data = numpy.random.randint(0, 255, (5, 2, 3, 301, 219), 'uint8')
            tif_w.save(final_image, compress=int(args["compression"]), photometric='minisblack')
print('done')
print('compression')
print(int(args["compression"]))