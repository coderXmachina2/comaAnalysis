import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

from astropy.io import fits
from astropy.wcs import WCS
from scipy import stats
from statsmodels import robust

from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

"""
These functions always return stuff.
"""


def zoomrescale(img, samplingfactor, mode="constant"):
    """
    Takes in an image. Resizes it with zoom. This is a wrapper function because I dont need ndimage in my main.
    """
    return ndimage.zoom(img, samplingfactor, mode="constant")


def upscale_repeat(img, upsampfact):
    """
    Takes in an image. Resizes it. It repeats the pixel by the scaling factor number.
    """
    # Calculate the new dimensions
    new_width = img.shape[1] * upsampfact
    new_height = img.shape[0] * upsampfact

    # Use NumPy's repeat function to upsample the image
    upsampled_image = np.repeat(np.repeat(img, upsampfact, axis=0), upsampfact, axis=1)
    return upsampled_image


def downscale_img(upsampled_img, downsampfact):
    """
    Takes in an upsampled image and downsamples it back to its original size.
    Averages the pixel values in each block of size `downsampfact` x `downsampfact`.
    """
    # Calculate the new dimensions
    new_width = upsampled_img.shape[1] // downsampfact
    new_height = upsampled_img.shape[0] // downsampfact

    # Chf the image is grayscale or coloreck i
    if len(upsampled_img.shape) == 3:
        # Color image
        num_channels = upsampled_img.shape[2]
        downsampled_image = np.zeros(
            (new_height, new_width, num_channels), dtype=upsampled_img.dtype
        )

        # Loop over each block and calculate the average value for each channel
        for i in range(new_height):
            for j in range(new_width):
                block = upsampled_img[
                    i * downsampfact : (i + 1) * downsampfact,
                    j * downsampfact : (j + 1) * downsampfact,
                ]
                downsampled_image[i, j] = block.mean(axis=(0, 1))
    else:
        # Shape Image = 2
        downsampled_image = np.zeros((new_height, new_width), dtype=upsampled_img.dtype)

        # Loop over each block and calculate the average value
        for i in range(new_height):
            for j in range(new_width):
                block = upsampled_img[
                    i * downsampfact : (i + 1) * downsampfact,
                    j * downsampfact : (j + 1) * downsampfact,
                ]
                downsampled_image[i, j] = block.mean()

    return downsampled_image
