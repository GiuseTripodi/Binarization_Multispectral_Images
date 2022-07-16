import shutil
import cv2
import glob
import os

'''
Run the following script to create the directory for the approach 1. The directory will hold all the images
where divided in normal images and GT.
'''

path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/ue2"
images_dir = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]

output_dir = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Test_images_approach_2"
for image_dir in images_dir:
    img_dir = os.path.join(path, image_dir)  # inside this dir there are all the channels
    channel = glob.glob(img_dir + "/F2s.png") # take only the second channel
    gt_channel = glob.glob(img_dir + "/*GT.png") # I am taking the GT image

    #copy images and change the name
    shutil.copyfile(channel[0], os.path.join(output_dir, f"org/F2s_{image_dir}.png"))
    shutil.copyfile(gt_channel[0], os.path.join(output_dir, f"gt/F2s_{image_dir}.png"))
