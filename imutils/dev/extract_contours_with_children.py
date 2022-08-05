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



def stack_extract_and_save_contours_with_children(input_filepath, output_folder, subsample=250):
    """

    :param input_filepath:
    :param output_folder:
    :param subsample:
    :return:
    """

    file_basename = os.path.splitext(os.path.basename(input_filepath))[0]
    print(file_basename)

    with tiff.TiffFile(input_filepath) as tif:
        for i, page in enumerate(tif.pages):
            if i % subsample != 0: continue
            img = page.asarray()

            #extract contours with children
            contours_with_children = imfunctions.extract_contours_with_children(img)

            for cnt_idx, cnt in enumerate(contours_with_children):
                # make the crop
                cnt_img = crop_image_from_contour(img, cnt)
                output_filepath = os.path.join(output_folder,
                                           file_basename + '_frame_' + str(i) + '_cnt_' + str(cnt_idx) + '.tiff')
                print(output_filepath)
                tiff.imwrite(output_filepath, cnt_img)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-i', '--i_path', help='input filepath', required=True)
    parser.add_argument('-o', '--o_path', help='output folder', required=True)
    parser.add_argument('-s', '--subsample', help='subsample', type=int, required=False)

    args = vars(parser.parse_args())
    input_filepath = args['i_path']
    output_folder = args['o_path']
    subsample = args['subsample']


    stack_extract_and_save_contours_with_children(input_filepath, output_folder, subsample=subsample)