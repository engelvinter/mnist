
import os
import numpy as np
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
from typing import List

def list_png_files(path : str) -> List[str]:
    png_files = [file for file in os.listdir(path) if file.endswith('.png')]
    return png_files

def read_png_files(path : str) -> np.ndarray:
    png_files = list_png_files(path)
    images = np.empty((len(png_files), 28, 28), dtype=np.uint8)
    for i, png_file in enumerate(png_files):
        file_path = os.path.join(path, png_file)
        image = Image.open(file_path).convert('L')  # Convert to grayscale ('L' mode)
        image_array = np.array(image)
        images[i] = image_array
    return images


SZ = 28 # images are SZ x SZ grayscale

def deskew(img : np.ndarray) -> np.ndarray:
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR)
    return img

def remove_empty_lines(img : np.ndarray) -> np.ndarray:
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
    rows, cols = img.shape
    colsPadding = (int(math.ceil((28 - cols) / 2.0)),int(math.floor((28 - cols) / 2.0)))
    rowsPadding = (int(math.ceil((28 - rows) / 2.0)),int(math.floor((28 - rows) / 2.0)))

    # np.lib.pad returns a new numpy array containing the padded image

    reshaped_img = np.lib.pad(img, (rowsPadding, colsPadding), 'constant')

    return reshaped_img

def symmetric(img : np.ndarray) -> np.ndarray:
    img1 = remove_empty_lines(img)
    img2 = reshape_20x20(img1)
    img3 = reshape_28x28(img2)
    return img3

def invert(images : np.ndarray) -> np.ndarray:
    (len, _, _) = images.shape
    for i in range(0, len):
        images[i] = cv2.bitwise_not(images[i])
    return images
    
def adjust_images(images : np.ndarray) -> np.ndarray:
    (len, _, _) = images.shape
    for i in range(0, len):
        img = deskew(images[i])
        images[i] = symmetric(img)
    return images

def scale_images(images : np.ndarray) -> np.ndarray:
    (len, rows, cols) = images.shape
    
    # The operation reshape returns either a new view or complete copy of numpy array of the image.

    scale_images = images.reshape((len, rows, cols, 1)).astype('float32') / 255
    return scale_images

def plot(img : np.ndarray) -> None:
    plt.imshow(img, cmap='grey')