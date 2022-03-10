#imports
import cv2
import tifffile as tiff
import glob
import os
# unet model is here (too)
from imutils.src.model import *
from natsort import natsorted


def unet_segmentation(img, model):
    """
    Segment an image based on the model loaded.
    Not sure if reshape and resize should be commented.

    :param img: image
    :param model: unet model
    :return: segmented_img
    """
    img_original_shape=img.shape
    # run U-Net network:
    img = cv2.resize(img, (256, 256))
    img = np.reshape(img, img.shape + (1,))
    img = np.reshape(img, (1,) + img.shape)

    img = img / 255
    results = model.predict(img)
    # reshape results
    results_reshaped = results.reshape(img_original_shape)
    # resize results
    results_reshaped = cv2.resize(results_reshaped, (w, h))
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
