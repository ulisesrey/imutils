import cv2
import tifffile as tiff
import matplotlib.pyplot as plt


def add_text(img, text, position, font, color):
    """
    position (x,y) tuple
    color, (R,G,B) tuple from  o to 254)
    """
    #img is your image in below line
    img = cv2.putText(img, text, position, font, 1, color, 2)
    #Display the image
    return img

path='/Users/ulises.rey/local_data/MondaySeminar3/2021-03-04_16-17-30_worm3_ZIM2051-channel-0-bigtiff_new_inv_MED_bg.btf kept stack_contrasted_inverted.tif'

font = cv2.FONT_HERSHEY_SIMPLEX
position = (300,100)
tif = tiff.TiffFile(path)

img = tif.pages[0].asarray()
plt.imshow(img)
plt.show()

img= add_text(img, 'Forward', position, font, (255,255,255))
plt.imshow(img)
plt.show()