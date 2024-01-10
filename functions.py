"""
    This module implements functions for the mnist digits
    There are functions for:
    * listing files in a directory
    * reading images from a directory
    * deskewing
    * removing empty lines around digit
    * resizing image into 20x20 or 28x28
    * normalize color data in image
    * invert image
    * plotting image
    There are also compound functions that uses the above functions:
    * symmetric
    * adjust_images

    Normally you only have to use the following two functions:
    * adjust_images and normalize_images
"""

import os
import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
from typing import List

SZ = 28 # images are SZ x SZ grayscale

def list_image_files(path : str) -> List[str]:
    """List jpg/png files in a specified path

    Parameters:
    path: the full path to the images

    Returns:
    an array of filenames
    """
    files = [file for file in os.listdir(path) if file.endswith('.png') or file.endswith('.jpg')]
    return files

def read_image_files(path : str) -> (List[str], np.ndarray):
    """Reads the 28x28 jpg/png images in the specified path and convert them to gray scale

    Parameters:
    path: the full path to the images

    Returns:
    a tuple containg a list of filenames and a numpy array (3 dim) of the gray-scale images
    """
    png_files = list_image_files(path)
    images = np.empty((len(png_files), SZ, SZ), dtype=np.uint8)
    for i, png_file in enumerate(png_files):
        file_path = os.path.join(path, png_file)
        image = Image.open(file_path).convert('L')  # Convert to grayscale ('L' mode)
        image_array = np.array(image)
        images[i] = image_array
    return (png_files, images)

def deskew(img : np.ndarray) -> np.ndarray:
    """Deskew an image.
       This is done by calculating skewness using moments,
       then applying an affine transformation to deskew the image.

    Parameters:
    image: the image as a numpy array (2 dim)

    Returns:
    a new numpy array (2-dim)
    """
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img

def remove_empty_lines(img : np.ndarray) -> np.ndarray:
    """Removes empty columns/rows (=black) in the border of the image

    Parameters:
    image: the image as a numpy array (2 dim)

    Returns:
    a new numpy array (2-dim)
    """
    # Slicing in numpy creates a new view object of the data
    while np.sum(img[0]) == 0:
        img = img[1:]

    while np.sum(img[:,0]) == 0:
        img = np.delete(img, 0, 1)

    while np.sum(img[-1]) == 0:
        img = img[:-1]

    while np.sum(img[:,-1]) == 0:
        img = np.delete(img, -1, 1)

    return img

def reshape_20x20(img : np.ndarray) -> np.ndarray:
    """Scales the provided image to 20x20

    Parameters:
    image: the image as a numpy array (2 dim)

    Returns:
    a new numpy array (2-dim)
    """
    rows, cols = img.shape

    # cv2.resize creates a new numpy array containing the resized image

    if rows > cols:
        factor = 20.0 / rows
        rows = 20
        cols = int(round(cols * factor))
        reshaped_img = cv2.resize(img, (cols,rows))
    else:
        factor = 20.0/cols
        cols = 20
        rows = int(round(rows*factor))
        reshaped_img = cv2.resize(img, (cols, rows))
    
    return reshaped_img

def reshape_28x28(img : np.ndarray) -> np.ndarray:
    """Reshapes the provided image. This is dowe by padding the image in both cols and rows.

    Parameters:
    img: the image as a numpy array (2-dim)

    Returns:
    a new numpy array (2-dim)
    """
    rows, cols = img.shape
    colsPadding = (int(math.ceil((SZ - cols) / 2.0)),int(math.floor((SZ - cols) / 2.0)))
    rowsPadding = (int(math.ceil((SZ - rows) / 2.0)),int(math.floor((SZ - rows) / 2.0)))

    # np.lib.pad returns a new numpy array containing the padded image

    reshaped_img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')

    return reshaped_img

def symmetric(img : np.ndarray) -> np.ndarray:
    """Removes any borders, resizes into 20x20 and then pads the image to get 28x28.
       The resulting image has symmetric borders.

    Parameters:
    img: the image as a numpy array (2-dim)

    Returns:
    a new numpy array (2-dim)
    """
    img1 = remove_empty_lines(img)
    img2 = reshape_20x20(img1)
    img3 = reshape_28x28(img2)
    return img3

def invert(images : np.ndarray) -> np.ndarray:
    """Inverts all the images (operation: bitwise not)

    Parameters:
    images: the images as a numpy array (3-dim/4-dim)

    Returns:
    a new numpy array of the same size (3-dim)
    """
    len = images.shape[0]
    for i in range(0, len):
        images[i] = cv2.bitwise_not(images[i])
    return images
    
def adjust_images(images : np.ndarray) -> np.ndarray:
    """Adjust the images by performing deskewing and making them symmetric.

    Parameters:
    images: the images as a numpy array (3-dim)

    Returns:
    a new numpy array of the same size (3-dim)
    """
    len = images.shape[0]
    for i in range(0, len):
        img = deskew(images[i])
        #img = images[i]
        images[i] = symmetric(img)
    return images

def normalize_images(images : np.ndarray) -> np.ndarray:
    """Normalize the gray scale image from color 0 - 255 to color 0.0 - 1.0

    Parameters:
    images: the images as a numpy array (3-dim)

    Returns:
    a new normalized numpy array of the same size (3-dim)
    """
    (len, rows, cols) = images.shape
    
    # The operation reshape returns either a new view or complete copy of numpy array of the image.

    scale_images = images.reshape((len, rows, cols, 1)).astype('float32') / 255
    return scale_images

def plot(img : np.ndarray) -> None:
    """Plots the provided image as gray scale

    Parameters:
    img: the image as a numpy array (2-dim)

    Returns:
    None
    """
    plt.imshow(img, cmap='grey')