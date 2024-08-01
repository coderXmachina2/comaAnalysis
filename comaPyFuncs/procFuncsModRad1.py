import pickle
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib

from comaPyFuncs import procFuncsMisc
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

#mask_annuli
#find_annulus_coordinates_spec

#mask_coordinates_with_specific_average
#find_annulus_coordinates_gen

#extract_pixels_within_circfatle
#extract_pixels_within_ellipse
#extract_circle_and_ellipse

def mask_annuli(image, center, radii, coeff=[], computation='mean' , mtype='', bkg=True, verboseDebug=False):
    """
def find_annul
    This is the main workhorse of annular models. But we are moving past annular models.
    
    Parameters:
    - image: NumPy array, the image to be masked.
    - center: Tuple of two integers (x, y), the center around which the annuli are defined.
    - radii: List of integers, defines the boundaries of each annulus. Successive elements
      define the inner and outer bounds of each annulus.
    - coeff: list modification coefficient experiment
    - computation:
    - mtype: sphere, ellipse
    - bkg: Boolean, if True, keep the values outside the mask the same as the image. 
           If False, set the values outside the mask to zero.

    Returns:
    - NumPy array, the image with masked annuli.
    - List of floats, the average values calculated for each annulus.
    """
    masked_image = np.copy(image)
    height, width = image.shape

    # Function to get pixels within the innermost circle
    if mtype == 'sphere':
        inner_pixel_values, inner_circle_coords = extract_pixels_within_circle(image, center, radii[0])
        max_radius = max(radii)
        min_radius = min(radii)
        if verboseDebug:
            print("Calculated Sphere:")
            print("Min rad:", min_radius )
            print("Max rad:", max_radius)
            print("Inner inner pixel values:", inner_pixel_values, 
                  "This test->", np.mean(inner_pixel_values), 
                  "Test val fin ->", (np.mean(inner_pixel_values)*coeff[0])+coeff[1]   )
            print("Inner inner circle coordinates", inner_circle_coords)
    elif mtype == 'ellipse':
        inner_pixel_values, inner_circle_coords = extract_pixels_within_ellipse(image, center, ellipse=(radii[0]))
        if verboseDebug:
            print("Calculated Ellipse:")
            print("Inner most max rad:", max_radius)
            print("Inner inner pixel values:", inner_pixel_values)
            print("Inner inner circle coordinates", inner_circle_coords)
    else:
        raise ValueError('Not a valid model type.')
        
    if inner_pixel_values:
        if len(coeff)> 0:
            if verboseDebug:
                print("Coeff activate!:", coeff)
            if computation == 'mean':
                inner_average_value = (np.mean(inner_pixel_values)*(coeff[0])) + coeff[1]
                if verboseDebug:
                    print("Inner average value: ->", inner_average_value)    #This value is printed but notadded
            elif computation == 'median':
                inner_average_value = (np.median(inner_pixel_values)*(coeff[0])) + coeff[1]
                if verboseDebug:
                    print("Inner median value:", inner_average_value)
        else:
            if computation == 'mean':
                inner_average_value = np.mean(inner_pixel_values)  
                if verboseDebug:
                    print("Inner average value", inner_average_value)
            if computation == 'median':
                inner_average_value = np.median(inner_pixel_values)  
                if verboseDebug:
                    print("Inner median value", inner_average_value)

        #It sets all those pixels to the inner average values
        for x, y in inner_circle_coords:
            masked_image[y, x] = inner_average_value

    rad_avg = [inner_average_value]
    # Mask each annulus with the average pixel value
    for i in range(len(radii) - 1): #This works with the circular model right... So the 
        if mtype == 'sphere':
            annulus_coords = find_annulus_coordinates_spec(masked_image, center, radii[i], radii[i+1])
            
        elif mtype == 'ellipse':
            #### This goes into that section ###
            pixelsA, coordsetA = extract_pixels_within_ellipse(image, center, ellipse=radii[i])
            pixelsB, coordsetB = extract_pixels_within_ellipse(image, center, ellipse=radii[i+1])
        
            annulus_coords = find_annulus_coordinates_gen(coordsetB, coordsetA)
            #### This goes into that section ###        
        
        if annulus_coords:
            pixel_values = [image[x, y] for x, y in annulus_coords]

            if len(coeff)> 0:
                if computation == 'mean':
                    average_value = (np.mean(pixel_values)*(coeff[0])) + coeff[1]
                    if verboseDebug:
                        print("xXMeanXx ->", average_value)
                elif computation == 'median':
                    average_value = (np.median(pixel_values)*(coeff[0])) + coeff[1]
                    if verboseDebug:
                        print("Median ->", average_value)
            else:
                if computation == 'mean':
                    average_value= np.mean(pixel_values) 
                    if verboseDebug:
                        print("Mean ->", average_value)
                elif computation == 'median':
                    average_value= np.median(pixel_values) 
                    if verboseDebug:
                        print("Median ->", average_value)
                    
            rad_avg.append(average_value)
            for x, y in annulus_coords:
                masked_image[x, y] = average_value

    if not bkg:
        if mtype == 'sphere':
            for x in range(height):
                for y in range(width):
                    if (x - center[0]) ** 2 + (y - center[1]) ** 2 > max_radius ** 2:
                        masked_image[x, y] = 0
        elif mtype == 'ellipse':
            for x in range(height):
                for y in range(width):
                    if not procFuncsMisc.is_within_ellipse(x, y, center, radii[-1][1], radii[-1][0], radii[-1][2]):
                        masked_image[x, y] = 0
            
    return masked_image, rad_avg

    ###########################################################################################################################
    #
    #
    #Experimental. Chat GPT recommended. Wait one. try youself first
def find_annulus_coordinates_spec_GPT(image, center, inner_params, outer_params):
    #Experimental. Chat GPT recommended. Wait one. try youself first
    """
    Calculate the pixel coordinates within a specified elliptical annulus.

    Parameters:
    - image: NumPy array, the image from which to calculate coordinates.
    - center: Tuple of two integers (x, y), the central point of the annulus.
    - inner_params: List of three values [a, b, c], where a is the semi-major axis, 
                    b is the semi-minor axis, and c is the inclination (in degrees) for the inner ellipse.
    - outer_params: List of three values [a, b, c], where a is the semi-major axis, 
                    b is the semi-minor axis, and c is the inclination (in degrees) for the outer ellipse.

    Returns:
    - List of tuples, each tuple is (x, y) coordinates of a pixel within the specified elliptical annulus.
    """
    #print("This is being used. GPT Function.")
    def is_within_ellipse(x, y, center, a, b, theta):
        theta = np.deg2rad(theta)
        cos_angle = np.cos(theta)
        sin_angle = np.sin(theta)
        
        xc = x - center[0]
        yc = y - center[1]

        xct = xc * cos_angle + yc * sin_angle
        yct = -xc * sin_angle + yc * cos_angle

        ellipse_eq = (xct ** 2) / (a ** 2) + (yct ** 2) / (b ** 2)
        return ellipse_eq <= 1

    inner_a, inner_b, inner_theta = inner_params
    outer_a, outer_b, outer_theta = outer_params
    
    coordinates = []
    for x in range(int(center[0] - outer_a), int(center[0] + outer_a + 1)):
        for y in range(int(center[1] - outer_b), int(center[1] + outer_b + 1)):
            if is_within_ellipse(x, y, center, outer_a, outer_b, outer_theta) and not is_within_ellipse(x, y, center, inner_a, inner_b, inner_theta):
                coordinates.append((x, y))

    return coordinates
    ###
    #Experimental. Chat GPT recommended. Wait one. try to implement youself first
    #
    #
    ###########################################################################################################################

def find_annulus_coordinates_spec(image, 
                                  center, 
                                  inner_radius, 
                                  outer_radius):
    """
    Calculate the pixel coordinates within a specified annulus. This is a bit thick,

    Parameters:
    - image: NumPy array, the image from which to calculate coordinates.
    - center: Tuple of two integers (x, y), the central point of the annulus.
    - inner_radius: Integer, the inner radius of the annulus.
    - outer_radius: Integer, the outer radius of the annulus.

    Returns:
    - List of tuples, each tuple is (x, y) coordinates of a pixel within the specified annulus.
    """
    #print("This is being used. Simple Function")
    #Debug
    #print( [(int(x), int(y)) for x in range(int(center[0]-outer_radius), int(center[0]+outer_radius+1))
    #        for y in range(int(center[1]-outer_radius), int(center[1]+outer_radius+1))
    #        if int(inner_radius**2) < (int(x-center[0])**2 + (y-center[1])**2) <= int(outer_radius**2)])

    return ( [(int(x), int(y)) for x in range(int(center[0]-outer_radius), int(center[0]+outer_radius+1))
            for y in range(int(center[1]-outer_radius), int(center[1]+outer_radius+1))
            if int(inner_radius**2) < (int(x-center[0])**2 + (y-center[1])**2) <= int(outer_radius**2)])

def find_annulus_coordinates_gen(outer_circle_salty_coords, 
                                 inner_circle_coords):
    """
    General funtion. Takes in outer circle and inner circle. Gets coordinates of the annulus. Compatible with sphere and or ellipse

    return
    """
    # Convert inner circle coordinates to a set for efficient lookup
    inner_set = set(inner_circle_coords)
    
    # Collect coordinates that are in the outer circle but not in the inner circle
    annulus_coordinates = [coord for coord in outer_circle_salty_coords if coord not in inner_circle_coords]

    return (annulus_coordinates)

def mask_coordinates_with_specific_average(image, 
                                           coordinates, 
                                           computation='mean'):
    """
    This is single mask. can be used iteratively. This function is stowed... Was used for experimentation...

    #It seems to have two very different outputs depending on whether it is mean or median. Which is not expected.

    Parameters:
    - image: NumPy array, the image to be masked.
    - coordinates: Tuple of two integers (x, y), the center around which the annuli are defined.
    - computation: List of integers, defines the boundaries of each annulus. Successive elements

    Returns:
    - NumPy array, the image with masked annuli.    
    """
    #print( [x for x in coordinates]  )
    # Extract pixel values at specified coordinates to calculate the average
    pixel_values = [image[x[0], x[1]] for x in coordinates]
    average_value = np.mean(pixel_values)

    if computation=='mean':
        # Create a mask of the same shape as the image, default to True
        average_value = 3        
    elif computation=='median':
        average_value = np.median(pixel_values)

    #print("Avg mask:", average_value)
    mask = np.ones(image.shape, dtype=bool)
    # Convert list of tuples to an array of indices
    coords_array = np.array(coordinates)
    
    # Set the coordinates in the mask to False (to mask them out)
    mask[coords_array[:, 0], coords_array[:, 1]] = False
    
    # Apply the mask to the image, setting masked out pixels to the calculated average
    masked_image = np.where(mask, image, average_value)
    
    return (masked_image)

#Used in one

def extract_pixels_within_circle(image, 
                                 center, 
                                 radius):
    """
    Returns a 1D list that are all the pixels within the specified circles
    """
    if(radius>0):
        # Create an empty list to store the pixel values
        pixels_within_circle = []
        
        # Calculate the boundary limits to avoid unnecessary calculations
        min_x = max(center[0] - radius, 0)
        max_x = min(center[0] + radius, image.shape[1])
        min_y = max(center[1] - radius, 0)
        max_y = min(center[1] + radius, image.shape[0])
    
        #ADD coordinates
        coordset = [ ]
        
        # Iterate over each pixel within the bounding box of the circle
        for y in range(int(min_y), int(max_y)):
            for x in range(int(min_x), int(max_x)):
                # Check if the pixel is within the circle
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2: #DIstance with respect to center...
                    pixels_within_circle.append(image[y, x])
                    coordset.append((x, y))
        
        return (pixels_within_circle, coordset)
    else:
        print("Invalid input")

                                  #semi_major, 
                                  #semi_minor, 
                                  #inc_angle_degrees):
def extract_pixels_within_ellipse(image, 
                                  center, 
                                  ellipse=()):

    """
    Extracts pixels within an ellipse defined by its center, semi-major and semi-minor axes, and inclination angle.

    Args:
    image (np.ndarray): 2D array representing the image.
    center (tuple): (x, y) coordinates of the ellipse's center.
    semi_major (int): Length of the semi-major axis.
    semi_minor (int): Length of the semi-minor axis.
    inc_angle_degrees (float): Inclination angle of the semi-major axis from the horizontal in degrees.

    Returns:
    list: Pixel values within the specified ellipse.
    """

    if(len(ellipse) > 0):
        pixels_within_ellipse = []
    
        # Convert inclination angle from degrees to radians
        theta = np.radians(ellipse[2])
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
    
        # Define bounds for the area to iterate over
        min_x = max(int(center[0] - ellipse[0]), 0)
        max_x = min(int(center[0] + ellipse[0]), image.shape[1])
        min_y = max(int(center[1] - ellipse[1]), 0)
        max_y = min(int(center[1] + ellipse[1]), image.shape[0])
    
        coordset=[]
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                # Apply the rotation matrix
                x_prime = cos_theta * (x - center[0]) + sin_theta * (y - center[1])
                y_prime = -sin_theta * (x - center[0]) + cos_theta * (y - center[1])
    
                # Check if the point is within the ellipse
                if (x_prime / ellipse[0]) ** 2 + (y_prime / ellipse[1]) ** 2 <= 1:
                    pixels_within_ellipse.append(image[y, x])
                    coordset.append((x, y))
        return (pixels_within_ellipse, coordset)
    else:
        print("Invalid input")


def extract_circle_and_ellipse(image, 
                               header, 
                               inbins=256, 
                               supT='', 
                               circrad=0,
                               ellipseparam=[],
                               logim=False):
    """
    Does both. A little redundant but why not.
    """

    if circrad != 0:
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        circle = Circle((center_x, center_y), 
                         circrad, color='red', 
                         fill=False, 
                         linestyle='--', 
                         linewidth=1.5,
                         alpha=0.55)
        circpixes = extract_pixels_within_circle(image, 
                                                 (int(center_x), int(center_y)),
                                                 circrad)
    if ellipseparam:
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        # Draw an ellipse at the center of the image
        ellipse = Ellipse((center_x, center_y), 
                          2*ellipseparam[0], 2*ellipseparam[1], 
                          angle=ellipseparam[2], edgecolor='red',
                          fill=False,
                          facecolor='none', linestyle='--', 
                          linewidth=1.5, alpha=0.55)
        ellipsepixes=extract_pixels_within_ellipse(image, 
                                      (int(center_x), int(center_y)), 
                                      ellipseparam[0], 
                                      ellipseparam[1], 
                                      ellipseparam[2])

    # Filter image with 3 sigma
    if logim:
        im = ax3.imshow(np.log(image), cmap='viridis')   

        cbar2 = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar2.set_label('Log Intensity', fontsize=16)
        ax3.set_title('Log Image', fontsize=16)
    else:
        mean = np.mean(image)
        std = np.std(image)
        filtered_image = np.clip(image, mean - 3 * std, mean + 3 * std)

        # Plot filtered image with WCS
        im = ax3.imshow(filtered_image, cmap='viridis')   

        ax3.set_title('Filtered Image (3 Sigma)', fontsize=16)

        cbar2 = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar2.set_label('Intensity', fontsize=16)
        cbar2.ax.tick_params(labelsize=16)

    if circrad != 0 and len(ellipseparam):
        return(circpixes, ellipsepixes)
    else:
        print("Invalid args")