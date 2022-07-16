import os
from matplotlib import image
from matplotlib import pyplot as plt

'''
This class represent a single image with all the wavelengths taken
'''
class image_:

    '''
    path: is the image directory
    '''
    def __init__(self, path):
        #take the name of the different wavelengths
        file = open(os.path.join(path, "listing_FX.txt"), "r")

        self.img = {}
        #load the images
        self.img["uv"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["vis1"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["vis2"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["vis3"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["ir1"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["ir2"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["ir3"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))
        self.img["ir4"] = image.imread(os.path.join(path, file.readline().strip().replace(".PNG", ".png")))

        self.n_channel = len(file.readline())

        self.shape = self.img["uv"].shape

    '''
    Function to plot bands from an image
    '''
    def plotBand(self, image_name: str, color: str):
        plt.figure(figsize=(30, 30))
        plt.imshow(self.img[image_name], cmap=color)
        plt.show()
        return

    def get_uv(self):
        return self.img["uv"]

    def get_vis1(self):
        return self.img["vis1"]

    def get_vis2(self):
        return self.img["vis2"]

    def get_vis3(self):
        return self.img["vis3"]

    def get_ir1(self):
        return self.img["ir1"]

    def get_ir2(self):
        return self.img["ir2"]

    def get_ir3(self):
        return self.img["ir3"]

    def get_ir4(self):
        return self.img["ir4"]

    def get_shape(self):
        return self.shape

