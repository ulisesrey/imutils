#imports
import cv2
import tifffile as tiff
import glob
import os
# unet model is here (too)
from imutils.src.model import *
from natsort import natsorted
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte

import pandas as pd
from natsort import natsorted
import matplotlib.image as mpimg
import os
import matplotlib.pyplot as plt



def unet_segmentation(img, model):
    """
    Segment an image based on the model loaded.
    Not sure if reshape and resize should be commented.

    :param img: image
    :param model: unet model
    :return: segmented_img
    """
    h, w = img.shape
    # run U-Net network:
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)

    img = img / 255
    results = model.predict(img)
    print(results.shape)
    # reshape results
    results_reshaped = results.reshape((256,256))
    print(results_reshaped.shape)
    # resize results
    results = cv2.resize(results_reshaped, (w, h)) # cv2 expects w, h in this order


    # multiply it by 255
    segmented_img = results * 255


    return segmented_img


def unet_segmentation_stack(input_filepath, output_filepath, weights_path):
    """"
    runs the unet_segmentation function for the stack
    """

    #load model and weights
    model=unet()
    print('loading weights..')
    model.load_weights(weights_path)

    with tiff.TiffFile(input_filepath, multifile=False) as tif,\
            tiff.TiffWriter(output_filepath, bigtiff=True) as tif_writer:
        for i, page in enumerate(tif.pages):
            img=page.asarray()
            #run network
            segmented_img=unet_segmentation(img, model)
            #write
            tif_writer.write(segmented_img, contiguous=True)

def testGenerator(test_path, target_size = (256,256),flag_multi_class = False, as_gray = True):
    """this function is duplicated from unet-master/data.py"""


    for i in natsorted(os.listdir(test_path)):
        print(i)
        img = io.imread(os.path.join(test_path, i),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,img.shape+(1,)) if (not flag_multi_class) else img
        img = np.reshape(img,(1,)+img.shape)
        yield img

def labelVisualize(num_class,color_dict,img):
    """this function is duplicated from unet-master/data.py"""
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255

def saveResult(save_path,npyfile,flag_multi_class = False,num_class = 2):
    """this function is duplicated from unet-master/data.py"""
    for i,item in enumerate(npyfile):
        img = labelVisualize(num_class,COLOR_DICT,item) if flag_multi_class else item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img_as_ubyte(img))


def predict_test_images(test_path, weights_path, save_path):
    """
    Creates the predictions for the images in the test_path based on the weights file
    Probably I could use the unet_segmentation() instead
    Args:
        test_path:
        weights_path:
        save_path:

    Returns:

    """
    model = unet()
    #generate test generator
    testGene = testGenerator(test_path=test_path)

    number_of_predictions = len(glob.glob(os.path.join(test_path, '*.tif*')))

    model.load_weights(weights_path)
    results = model.predict_generator(testGene,number_of_predictions,verbose=1)

    # save_path=os.path.join(project_path, test_folder+'_predictions')
    #print(save_path)
    try: os.mkdir(save_path)
    except: print('folder already exists')

    saveResult(save_path, results)

    # return results

def training_results_figure(training_folder):
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(20, 5), nrows=1, ncols=3)
    fig.subplots_adjust(wspace=0.5, hspace=0)
    try:
        df = pd.read_csv(os.path.join(training_folder, 'history.csv'))
        df = df.iloc[:, 1:]
        min_val_loss = round(df['val_loss'].min(), 3)
        max_val_accuracy = round(df['val_accuracy'].max(), 3)

        df[['val_loss', 'loss']].plot(ax=ax1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Min val loss: ' + str(min_val_loss))
        ax1.set_ylim(0, 1)
        # plt.show()
        df[['val_accuracy', 'accuracy']].plot(ax=ax2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Max val accuracy: ' + str(max_val_accuracy))
    except:
        print('no history file for this model: ', training_folder)

    # plt.show()
    img = mpimg.imread(os.path.join(training_folder, 'test_predictions', '1_predict.png'))
    ax3.imshow(img)
    ax3.set_axis_off()

    return fig, (ax1, ax2, ax3)

# training_results_path='/Volumes/scratch/neurobiology/zimmer/ulises/code/unet-master/data/worm_segmentation_all_worms/training_results/*'
#
# training_folders=glob.glob(training_results_path)
#
# for training_folder in natsorted(training_folders):
#     fig, (ax1, ax2, ax3) = training_results_figure(training_folder)
#     fig.suptitle(os.path.basename(training_folder))
#     plt.show()

# input_filepath='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20220127/data/worm6/2022-01-27_22-09-19_worm6-channel-0-behaviour-/2022-01-27_22-09-19_worm6-channel-0-behaviour-bigtiff_AVG_background_substracted.btf'
# output_filepath='/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20220127/data/worm6/2022-01-27_22-09-19_worm6-channel-0-behaviour-/2022-01-27_22-09-19_worm6-channel-0-behaviour-bigtiff_AVG_background_substracted_unet_segmented.btf'
# weights_path='/Volumes/scratch/neurobiology/zimmer/ulises/code/unet-master/data/2022_04_24_worm_segmentation_all_worms_good_background/training_results/2626248_1_w_validation_2batchsize_500steps_100epochs_patience/unet_master.hdf5'
# unet_segmentation_stack(input_filepath, output_filepath, weights_path)