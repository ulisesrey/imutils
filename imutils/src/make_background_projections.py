import tifffile
import numpy as np
import os
import glob
from imutils.src.imfunctions import stack_z_projection

if __name__ == "__main__":
    main_path="/Volumes/scratch/neurobiology/zimmer/active_sensing/zim06/recordings_to_analyze/josefine_neurite"
    background_files = os.path.join(main_path, "*/data/*worm*/*background*/*background*/*background*.tif")
    print(background_files)
    for background_filepath in glob.glob(background_files):
        #get the path to the background file without the filename
        print(background_filepath)
        output_path = os.path.join(os.path.dirname(background_filepath), 'AVG_background_projection.tif')
        print(output_path,"\n")
        #stack_z_projection(background_filepath, output_path, projection_type='mean', axis=0)