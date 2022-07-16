# for data load
import os
import torch
# for reading and processing images
import imageio
from matplotlib import image
from PIL import Image
from matplotlib import pyplot as py

# for visualizations
import matplotlib.pyplot as plt
import numpy as np # for using np arrays
import glob
import random

# for bulding and running deep learning model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, metrics
import pandas as pd




# code from: https://colab.research.google.com/github/VidushiBhatia/U-Net-Implementation/blob/main/U_Net_for_Image_Segmentation_From_Scratch_Using_TensorFlow_v4.ipynb#scrollTo=dwjr7eZQEK36

def LoadData(path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively

    """
    # Read the images folder like a list
    image_dataset = os.listdir(path1)
    mask_dataset = os.listdir(path2)

    # Make a list for images and masks filenames
    orig_img = []
    mask_img = []
    for file in image_dataset:
        orig_img.append(file)
    for file in mask_dataset:
        mask_img.append(file)

    # Sort the lists to get both of them in same order (the dataset has exactly the same name for images and corresponding masks)
    orig_img.sort()
    mask_img.sort()

    return orig_img, mask_img


def pad_image_to_tile_multiple(image3, tile_size, padding="CONSTANT"):
    imagesize = image3.shape

    target_height = imagesize[0] - (imagesize[0] % tile_size[0]) + tile_size[0]
    target_width = imagesize[1] - (imagesize[1] % tile_size[1]) + tile_size[1]

    add_height = target_height - imagesize[0]
    add_width = target_width - imagesize[1]

    return np.pad(image3, ((0, add_height), (0, add_width)), "constant", constant_values=0), (target_height, target_width)


# code adapted from: https://stackoverflow.com/questions/38235643/getting-started-with-tensorflow-split-image-into-sub-images

def split_image(image3, tile_size):
    image_shape = tf.shape(image3)
    tile_rows = tf.reshape(image3, [image_shape[0], -1, tile_size[1]])
    serial_tiles = tf.transpose(tile_rows, [1, 0, 2])
    return tf.reshape(serial_tiles, [-1, tile_size[1], tile_size[0]])


def unsplit_image(tiles4, image_shape):
    tile_width = tf.shape(tiles4)[1]
    serialized_tiles = tf.reshape(tiles4, [-1, image_shape[0], tile_width])
    rowwise_tiles = tf.transpose(serialized_tiles, [1, 0, 2])
    return tf.reshape(rowwise_tiles, [image_shape[0], image_shape[1]])


'''
This function get the images and the training mask, and fro every every image, mask create
a block of size [target_shape_img], and add this block to the training array X

'''
def PreprocessData(img, mask, target_shape_block, path1, path2):
    '''
    Processes the images and mask present in the shared list and path, split the images
    and the masks in the target_shape_block indicated, and add the block to the output list

    target_shape_block is a 2 dimensional array, with the expected blok size
    '''

    X = {}
    y = {}
    for file in img:
        print(file)
        index = img.index(file)
        single_mask_ind = mask[index]

        # get images and mask
        # single_image = imageio.imread(os.path.join(path1, file))
        single_image = image.imread(os.path.join(path1, file))
        single_image = single_image / 255

        # single_mask = imageio.imread(os.path.join(path2, single_mask_ind))
        single_mask = image.imread(os.path.join(path2, single_mask_ind))
        single_mask = single_mask / 255

        # split the images, both images and mash have the same size

        single_image = pad_image_to_tile_multiple(single_image, target_shape_block)
        single_mask = pad_image_to_tile_multiple(single_mask, target_shape_block)

        single_image = tf.convert_to_tensor(single_image,
                                            dtype=tf.float32)  # this is converting the the numpy array further
        single_mask = tf.convert_to_tensor(single_mask, dtype=tf.float32)

        tiles_image = split_image(single_image, target_shape_block)
        tiles_mask = split_image(single_mask, target_shape_block)

        for tile in range(len(tiles_image)):
            X[file.replace(".png", f"_{tile}")] = tiles_image[tile]
            y[file.replace(".png", f"_{tile}")] = tiles_mask[tile]

    # create a proper dataset
    m = len(X.keys())
    i_h = target_shape_block[0]
    i_w = target_shape_block[1]
    X_ = np.zeros((m, i_h, i_w), dtype=np.float)
    y_ = np.zeros((m, i_h, i_w), dtype=np.int32)

    for i in range(m):
        X_[i] = X[list(X.keys())[i]]
        y_[i] = y[list(y.keys())[i]]

    return X_, y_


'''
This function get the image in the path and for every image takes the different channels of the image, then 
create different block of size shape_block, and concatenate random all the block for the same image but coming 
from different channels
'''
def concatenate_channel(path , shape_block, image_shape):
    images_dir = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    #test
    print(images_dir)

    for image_dir in images_dir:
        print(image_dir)
        first = True
        img_dir = os.path.join(path, image_dir) # inside this dir there are all the channels
        channels = glob.glob(img_dir + "/F*s.png") # I am not taking the GT image
        gt_channel = glob.glob(img_dir + "/*GT.png") # I am taking the GT image

        for channel in channels:
            ch_img = image.imread(channel)
            gt_img = image.imread(gt_channel[0])

            ch_img, image_shape_pad = pad_image_to_tile_multiple(ch_img, shape_block)
            gt_img, _ = pad_image_to_tile_multiple(gt_img, shape_block)

            gt_til = split_image(gt_img, shape_block)
            til = split_image(ch_img, shape_block)
            if first:
                blocks = til
                gt_blocks = gt_til
                first = False
            else:
                blocks = tf.concat((blocks, til), axis=0)
                gt_blocks = tf.concat((gt_blocks, gt_til), axis=0)

        #get the indices and shuffle it
        indices = tf.range(start=0, limit=tf.shape(blocks)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        #shuffle the tiles
        blocks = tf.gather(blocks, shuffled_indices)
        gt_blocks = tf.gather(gt_blocks, shuffled_indices)


        #create the images
        b = 0
        n_blocks_for_image = round((image_shape_pad[0] * image_shape_pad[1]) / (shape_block[0] * shape_block[1]))

        Test_images_approach_1 = "/mnt/1028D91228D8F7A4/universita/Magistrale/Document Analysis/Esercizi/Assignment 2/Binarization_Multispectral_Images/Test_images_approach_1"
        for i in range(len(channels)):
            tiles = tf.slice(blocks, begin=[b, 0, 0], size=[n_blocks_for_image, shape_block[0], shape_block[1]])
            img = unsplit_image(tiles, image_shape_pad)
            plt.imsave(f"{Test_images_approach_1}/org/{image_dir}_FN{i}.png", img)

            #Crete the GT image
            gt_tiles = tf.slice(gt_blocks, begin=[b, 0, 0], size=[n_blocks_for_image, shape_block[0], shape_block[1]])
            gt_img = unsplit_image(gt_tiles, image_shape_pad)
            print(gt_img)
            plt.imsave(f"{Test_images_approach_1}/gt/{image_dir}_FN{i}.png", gt_img, cmap="gray")

            b += n_blocks_for_image




