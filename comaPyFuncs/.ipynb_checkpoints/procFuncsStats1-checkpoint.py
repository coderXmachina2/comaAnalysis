import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.wcs import WCS
from scipy import stats
from statsmodels import robust

from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

from scipy.signal import correlate2d
from skimage import io, color


"""
These functions do statistical analysis.
"""

#autocovariance_2d
#img_diag

def autocovariance_2d_a(image):
    """
    Computes the autocovariance of a 2D image.
    Calculates the mean. Preserves the 
    Subtracts the image with the mean to get an image of a mean of 0
    
    Parameters:
    image (numpy.ndarray): Input 2D image.
    
    Returns:
    numpy.ndarray: 2D autocovariance matrix.
    """
    # Ensure the input is a 2D array
    assert len(image.shape) == 2, "Input image must be a 2D array"

    # Subtract the mean of the image
    image_mean = np.mean(image)
    image_zero_mean = image - image_mean

    # Get the dimensions of the image
    rows, cols = image.shape

    # Initialize the autocovariance matrix... Generates a field of 0s
    autocov_matrix = np.zeros((rows, cols))

    # Compute the autocovariance FOR EACH SHIFT
    for i in range(rows):
        for j in range(cols):
            shifted_image = np.roll(np.roll(image_zero_mean, i, axis=0), j, axis=1)
            autocov_matrix[i, j] = np.mean(image_zero_mean * shifted_image) #Taking the mean of the product of the 0 mean image and the shifted 
    
    return autocov_matrix

def autocovariance_2d_b(image):
    """
    Computes the autocovariance of a 2D image.
    
    Parameters:
    image (numpy.ndarray): Input 2D image.
    
    Returns:
    numpy.ndarray: 2D autocovariance matrix.
    """
    # Ensure the input is a 2D array
    assert len(image.shape) == 2, "Input image must be a 2D array"

    image_mean = np.mean(image)
    image_zero_mean = image - image_mean
    
    # Step 3: Compute the 2D autocovariance
    autocovariance = correlate2d(image_zero_mean, image_zero_mean, mode='full')
    
    # Step 4: Normalize the autocovariance
    autocovariance /= (image.shape[0] * image.shape[1])
        
    return autocovariance 

def img_diagnosat(image, 
                  integarround= 10,
                  verbose=False):
    """
    Returns statistics
    """
    
    # Create subplots
    if (verbose):
        print("Max:", np.round( np.max(image), 10),
              "Min:", np.round( np.min(image), 10),
              "Range:", np.round( np.max(image)+np.min(image), 10))
        print("Mean \u00B1 std:", np.round(np.mean(image), 10), "\u00B1", np.round( np.std(image), 10   )) 
        print("Median \u00B1 median absolute deviation:", np.round(np.median(image), 10), "\u00B1", np.round(robust.mad(image.ravel()), 10))    
        print("Variance:", np.round( np.std(image)*np.std(image), 10))
    
    return({"Max":   np.round(np.max(image),  10),
            "Min":   np.round(np.min(image),  10),
            "Mean":  np.round(np.mean(image), 10), 
            "Median":np.round(np.median(image), 10),
            "Std":np.round(np.std(image), 10), 
            "MAD": np.round(robust.mad(image.ravel()), 10)}  )