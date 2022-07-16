'''
This script contain the code to take the the output images from the model and to compute the
metrics on that. The images used to do that are in the comparation_test_images dir
'''


import cv2
import matplotlib
from matplotlib import image
import imageio

import os
from Binarization.measures.evaluation_measures import Measure

def comparation_test():
    # load images
    path = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Data/comparation_test_images" # in this path there are the directory of the different epoch
    for epochs in os.listdir(path):
        print(f"Result after {epochs}\n")
        approaches = os.listdir(os.path.join(path, epochs)) # this file contain the name of the directory for each approach used
        for approach in approaches:
            print(f"Approach: {approach}\n")
            #inide approach there are 2 directory: images and gt
            path_approach = os.path.join(path, epochs, approach)
            files = os.listdir(os.path.join(path_approach, "images"))
            #get all the file
            img = {}
            gt_img = {}
            for file in files:

                img[file] = cv2.cvtColor(imageio.imread(os.path.join(path_approach, "images", file)), cv2.COLOR_BGR2GRAY )
                gt_img[file] = cv2.cvtColor(imageio.imread(os.path.join(path_approach, "gt", file.replace('.png', '_GT.png'))), cv2.COLOR_BGR2GRAY)

                img[file] = img[file] // 175
                gt_img[file] = gt_img[file] // 175



            # compute the measures
            f_measure = 0
            pf_measure = 0
            psnr = 0
            drd = 0

            for i in img.keys():
                m = Measure(img[i], gt_img[i])
                m.f_measure()
                m.p_f_measure()
                m.psnr()
                m.drd()

                f_measure += m.getFmeasure()
                pf_measure += m.getPFmeasure()
                psnr += m.getPSNR()
                drd += m.getDRD()

            f_measure /= len(img)
            pf_measure /= len(img)
            psnr /= len(img)
            drd /= len(img)

            # Results
            # print measures
            print("\n")
            print(f"f measure: {f_measure}")
            print(f"Pf measure: {pf_measure}")
            print(f"PSNR: {psnr}")
            print(f"DRD: {drd} \n")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    comparation_test()