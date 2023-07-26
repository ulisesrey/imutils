#Test to see if extract_contours_with_children(img) returns None when img has no contours with children

# It looks like returns an empty list []
import tifffile as tiff
import matplotlib.pyplot as plt
import cv2


def extract_contours_with_children_local(img):
    """
    Find the contours that have a children in the given image and return them as list

    Parameters:
    -----------
    img, np.array
    Returns:
    -----------
    contours that have a children, list of contours with children
    Important does not return the children, only the contour that has children.
    """

    #important, findCountour() has different outputs depending on CV version!
    cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(cnts))
    # print(hierarchy)
    contours_with_children = []
    for cnt_idx, cnt in enumerate(cnts):
        # draw contours with children: last column in the array is -1 if an external contour, column 2 is different than -1 meaning it has children
        if hierarchy[0][cnt_idx][3] == -1 and hierarchy[0][cnt_idx][2] != -1:
            contours_with_children.append(cnt)
            # not needed, do it outside this function
            # get coords of boundingRect
            # x,y,w,h = cv2.boundingRect(cnt)
            # make the crop
            # cnt_img=img[y:y+h,x:x+w]
    return contours_with_children

binary_input_filepath = '/Volumes/scratch/neurobiology/zimmer/ulises/wbfm/20220729_12ms/data/ZIM2156_worm12/2022-07-29_18-08_ZIM2156_worm12_Ch0-BH/2022-07-29_18-08_ZIM2156_worm12_Ch0-BHbigtiff_AVG_background_subtracted_normalised_unet_segmented_weights_5358068_1_005mask.btf'

with tiff.TiffFile(binary_input_filepath) as tif_binary:
    for i, page in enumerate(tif_binary.pages):
        img=page.asarray()
        plt.imshow(img)
        extract_contours_with_children_local(img)

