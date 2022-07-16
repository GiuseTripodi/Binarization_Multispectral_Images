from Data.helper_function import *
import cv2
from Data.image_ import image_

'''
Run the following script to create the directory for the approach 1. The directory will hold all the images
where are concatenated the blocks.
'''

# read image
path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/ue2/z27"

img = image_(path)
image_shape = img.get_shape()
path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/ue2/"
shape_block = (64, 64)
concatenate_channel(path, shape_block, image_shape)