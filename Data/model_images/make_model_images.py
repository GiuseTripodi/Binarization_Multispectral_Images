import os
import shutil

'''
This script create the directories useful to train the conv network
'''

name = "F2s"  # name of the kind of file to move

gt_dir_path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/model_images/gt"
org_dir_path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/model_images/org"

dir_path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/ue2"
for directories in os.listdir(dir_path):
    if os.path.isdir(os.path.join(dir_path, directories)):
        print(f"directory: {directories}")
        # move the file selected
        shutil.copy(os.path.join(dir_path, directories, name + ".png"), os.path.join(org_dir_path,  f"{name}_{directories}.png"))
        #move the gt
        shutil.copy(os.path.join(dir_path, directories, directories + "GT.png"), os.path.join(gt_dir_path, f"{name}_{directories}_GT.png"))


