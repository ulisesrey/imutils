from imutils.src import imfunctions
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2
import os


def crop_image_from_contour(img, contour):
    """crop the image based on a contour
    Parameters:
    -----------
    img, img
    contour, contour object
    returns:
    cnt_img
    """
    x, y, w, h = cv2.boundingRect(contour)
    # make the crop
    cnt_img = img[y:y + h, x:x + w]

    return cnt_img


def stack_extract_and_save_contours_with_children(binary_input_filepath, raw_input_filepath, output_folder, crop=False,
                                                  subsample=250):
    """

    :param binary_input_filepath: image from where the contours will be extracted
    :param raw_input_filepath: image that will be cropped based on the binary contours (can be the same as binary_input_filepath)
    :param output_folder: folder where to save the images
    :param subsample: subsample rate to not to save all the images.
    :return: None
    """

    file_basename = os.path.splitext(os.path.basename(raw_input_filepath))[0]
    print(file_basename)

    with tiff.TiffFile(binary_input_filepath) as tif_binary, tiff.TiffFile(raw_input_filepath) as tif_raw:
        for i, page in enumerate(tif_binary.pages):
            if i % subsample != 0: continue
            img = page.asarray()

            # extract contours with children
            contours_with_children = imfunctions.extract_contours_with_children(img)

            for cnt_idx, cnt in enumerate(contours_with_children):
                # get the raw image
                raw_img = tif_raw.pages[i].asarray()
                # if crop == True make the crop on the raw image
                if crop==True:
                    raw_img = crop_image_from_contour(raw_img, cnt)

                output_filepath = os.path.join(output_folder,
                                               file_basename + '_frame_' + str(i) + '_cnt_' + str(cnt_idx) + '.tiff')
                print(output_filepath)
                tiff.imwrite(output_filepath, raw_img)
    return None


if __name__ == "__main__":
    # run with array_job_directories_extract_contours.sh
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-bi', '--bi_path', help='binary input filepath', required=True)
    parser.add_argument('-ri', '--ri_path', help='raw input filepath', required=True)
    parser.add_argument('-o', '--o_path', help='output folder', required=True)
    parser.add_argument('-c', '--crop', help='crop', required=False)
    parser.add_argument('-s', '--subsample', help='subsample', type=int, required=False)

    args = vars(parser.parse_args())
    binary_input_filepath = args['bi_path']
    raw_input_filepath = args['ri_path']
    output_folder = args['o_path']
    crop = args['crop']
    subsample = args['subsample']

    stack_extract_and_save_contours_with_children(binary_input_filepath, raw_input_filepath, output_folder, crop,
                                                  subsample=subsample)
