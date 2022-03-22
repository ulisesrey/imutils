import cv2
import tifffile as tiff
import matplotlib.pyplot as plt
import pandas as pd




path='/Users/ulises.rey/local_data/MondaySeminar3/2021-03-04_16-17-30_worm3_ZIM2051-channel-0-bigtiff_new_inv_MED_bg.btf kept stack_contrasted_inverted_90degrees_left.tif'
output_path='/Users/ulises.rey/local_data/MondaySeminar3/2021-03-04_16-17-30_worm3_ZIM2051-channel-0-bigtiff_new_inv_MED_bg.btf kept stack_contrasted_inverted_90degrees_left_annotated.tif'

path_motor_state='/Users/ulises.rey/local_data/worm3_wbfm_PCA/beh_annotation_16_subsamples_timeseries_with_long_reversals.csv'
# #'/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/worm3/beh_annotation_16_subsamples_timeseries_with_long_reversals.csv'
motor_state_df=pd.read_csv(path_motor_state)
#subsample motorstate
#motor_state_df=motor_state_df.loc[::2,:]

font = cv2.FONT_HERSHEY_SIMPLEX
position = (300,165)

color_dict = {'forward': (0,0,255),
              'reversal': (255,0,0),
              'sustained reversal': (255,0,0),
              'ventral turn': (255,140,0),
              'dorsal turn': (255,140,0)}

text_dict = {'forward': 'Forward',
              'reversal': 'Reversal',
              'sustained reversal': 'Reversal',
              'ventral turn': 'Turn',
              'dorsal turn': 'Turn'}

with tiff.TiffFile(path) as tif, tiff.TiffWriter(output_path) as tif_writer:
    for idx, page in enumerate(tif.pages):
        img = page.asarray()
        # i had to multiply the index because the behaviour was annotated on a subsampled image
        idx=idx*2

        # convert the img to RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        #print(motor_state_df.loc[idx,'state'])
        # Try statement in case idx is outside of the dataframe range
        try:
            text = text_dict[motor_state_df.loc[idx,'state']]
            color = color_dict[motor_state_df.loc[idx,'state']]
            #print('text is :', text, 'color is :', color)
            # add text
            img = cv2.putText(img, text, position, font, 1.5, color, 4)

        except: print('idx not in image')
        # write
        tif_writer.write(img)# not required: photometric='rgb')
        #break