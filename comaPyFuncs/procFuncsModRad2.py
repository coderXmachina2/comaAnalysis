import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib

from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from skimage.draw import ellipse_perimeter, circle_perimeter

"""
These functions always return stuff. These functions are related to Radial Models.
"""

def generate_shape_and_get_border_pixels(image, mtype='', params=((), [])):
    """
    Generates a sphere or ellipse on the image and returns the border pixel locations and their intensities.
    
    Args:
    image (np.ndarray): Input 100x100 image.
    shape_type (str): Type of shape ('sphere' or 'ellipse').
    center (tuple): Center coordinates of the shape (centerx, centery).
    size (tuple): Size of the shape. For 'sphere', this is (radius,). For 'ellipse', this is (semi-major axis, semi-minor axis, angle).
    
    Returns:
    border_pixels (list): List of (x, y) coordinates of the border pixels.
    border_intensities (list): List of intensities of the border pixels.
    """
    
    # Copy the image to draw on
    img = image.copy()
    
    if mtype == 'sphere':
        radius = params[1]
        rr, cc = circle_perimeter(params[0][0], params[0][1], params[1]) #v
    
    elif mtype == 'ellipse':
        #print("This is params:", params)
        semi_major_axis, semi_minor_axis, angle = params[1]
        rr, cc = ellipse_perimeter(params[0][0], #coord center
                                   params[0][1], #coord center
                                   params[1][1], #This should be semi minor
                                   params[1][0], #This should be semi major
                                   orientation=np.deg2rad(params[1][2])) #Major axis orientation in clockwise direction as radians.
    else:
        raise ValueError("Invalid shape type. Choose 'sphere' or 'ellipse'.")
    
    # Ensure the coordinates are within the image bounds
    valid_idx = (rr >= 0) & (rr < image.shape[0]) & (cc >= 0) & (cc < image.shape[1])
    rr, cc = rr[valid_idx], cc[valid_idx]

    border_pixels = list(zip(rr, cc))
    border_intensities = [img[r, c] for r, c in border_pixels]
    
    return (border_pixels, border_intensities)


def set_mean_pixel_values(image, 
                          data, 
                          mType='',
                          blanked=False):
    """
    Sets the pixel values at the specified coordinates to the mean of their respective values.

    Parameters:
    image (np.ndarray): The input image.
    data (list): A list of tuples where each tuple contains a list of coordinates and a list of pixel values.
                 Format: [([coord1, coord2, ...], [value1, value2, ...]), ...]

    Returns:
    np.ndarray: The modified image.
    """
    #Copy the image
    imgcpy = image.copy()

    #Declare an empty set
    all_coords = set()

    for coords, values in data:
        mean_value = np.mean(values)
        for x, y in coords:
            imgcpy [x, y] = mean_value #This is fine because we tried it earlier
            all_coords.add((x, y))

    # If blanked is True, set all other pixel values to 0
    if blanked:
        imgcpy_mask = np.zeros_like(image)
        for x, y in all_coords:
            imgcpy_mask[x, y] = imgcpy[x, y]
        imgcpy = imgcpy_mask

    return imgcpy

def interpolate_shape(image, methodint='linear', verbose=False):
    # Get the coordinates of the non-zero points (points on ellipses)
    non_zero_coords = np.array(np.nonzero(image)).T
    non_zero_values = image[non_zero_coords[:, 0], non_zero_coords[:, 1]]
    
    # Define the grid for interpolation
    grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    
    if verbose:
        print("Non-zero coordinates:", non_zero_coords)
        print("Grid X shape:", grid_x.shape, "Grid Y shape:", grid_y.shape)

    # Perform interpolation
    interpolated_image = griddata(points=non_zero_coords,
                                  values=non_zero_values,
                                  xi=(grid_x, grid_y),
                                  method=methodint)

    if verbose:
        print("Interpolation method:", methodint)

    image_with_interpolation = image.copy()
    zero_coords = np.where(image == 0)

    interpolated_image [np.isnan(interpolated_image )] = 0

    return interpolated_image.T