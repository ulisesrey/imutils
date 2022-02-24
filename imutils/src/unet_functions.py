#imports
import cv2
import tifffile as tiff

# unet model is here (too)
from imutils.src.model import *


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
