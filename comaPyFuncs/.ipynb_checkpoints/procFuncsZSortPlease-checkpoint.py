import importlib
import glob
import importlib
import skimage
import skimage.filters
import random 
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

from matplotlib import pyplot as plt
from astropy.io import fits as pyfits
from astropy.wcs import WCS
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from skimage.draw import ellipse_perimeter, circle_perimeter

#procFuncsZSortPlease.
def fit_ellipse_params(list_params):
    x_data = np.arange(len(list_params))

    igdata = [max(list_params), 1, np.median(x_data)]

    parmsSemiMaj, covarsemiMaj = curve_fit(logis_func,
                                           x_data,
                                           list_params,
                                           p0=igdata)
    L, k, x0 = parmsSemiMaj
    fitted_axis = logis_func(x_data, L, k, x0)

    return ([int(x) for x in fitted_axis])


def logis_func(x, L, k, x0):
    return(L / (1 + np.exp(-k*(x-x0))))

def calcdist(  point1,  point2):
    """
    Calculates distance between two points. Cartesian Coordinates.
    """
    return((math.sqrt((point1[0] - point2[0])**2) + ((point1[1] - point2[1])**2)))

def set_pixels(image, coordinates):#, sub_val=0.0001): #initial sub_val 0.0001
    """
    Set specified pixel coordinates in an image to 1.

    Parameters:
    image (np.ndarray): Original image as a numpy array.
    coordinates (list): List of tuples containing the pixel coordinates to set to 1.

    Returns:
    np.ndarray: Modified image with specified pixels set to 1.
    """

    imgc= image.copy()
    sub_val = 10*np.std(imgc)
    
    # Set the specified coordinates to 1
    for coord in coordinates:
        imgc[coord] =  sub_val
    
    return imgc

def process_derivative(inDeriv, threshF=0.01, verboseH = False):
    """
    This is a function that returns us the critical/ neutral point

    #Critical points = Neutral points.
    #Even the image is largely 0. Exponential is 0.
    """
    Ix, Iy = inDeriv
    
    # Calculate thresholds for Ix and Iy to consider them effectively zero
    Ix_threshold = threshF * np.std(Ix) + np.mean(Ix)
    Iy_threshold = threshF * np.std(Iy) + np.mean(Iy)

    if (verboseH):
        print("Ix threshold:", Ix_threshold)
        print("Iy threshold:", Iy_threshold)
    
    # Initialize image copy with same shape as Ix and Iy
    false_colour = np.full_like(Ix, fill_value=1000)  # Assuming 255 is the default value for other points
    
    # Find locations where Ix and Iy are effectively zero (x, y)
    neutral_points = np.where((np.abs(Ix) <= Ix_threshold) & (np.abs(Iy) <= Iy_threshold))

    if (verboseH):
        print("\nN neutral points:")
        print("Where size (2):", len(neutral_points))
        print(len(neutral_points[0]), len( neutral_points[1] ))
        print("Sample Y vals:", neutral_points[0][:10])
        print("Sample X vals:", neutral_points[1][:10])

        print("Proof of neutral point. These points are below threshold:")
        print("Ix value:", Ix[neutral_points[0][0]][neutral_points[1][0]], "of y:", neutral_points[0][0], "x:", neutral_points[1][0])
        print("Iy value:", Iy[neutral_points[0][0]][neutral_points[1][0]], "of y:", neutral_points[0][0], "x:", neutral_points[1][0])
        
        #print("If this coordset was switched.  (x first) Where prints y first.") #
        #print("Ix value:", Ix[neutral_points[1][0]][neutral_points[0][0]], "of y:", neutral_points[1][0], 
        #      "x:", neutral_points[0][0])
        #print("Iy value:", Iy[neutral_points[1][0]][neutral_points[0][0]], "of y:", neutral_points[1][0], "x:", neutral_points[0][0])

        print("Proof of neutral point. These points are well below threshold:")
        print("Ix value:", Ix[neutral_points[0][1]][neutral_points[1][1]], "of y:", neutral_points[0][1], "x:", neutral_points[1][1])
        print("Iy value:", Iy[neutral_points[0]][1][neutral_points[1][1]], "of y:", neutral_points[0][1], "x:", neutral_points[1][1])

        #print("If this coordset was switched. (x first) Where prints y first.")
        #print("Ix value:", Ix[neutral_points[1][1]][neutral_points[0][1]], "of y:", neutral_points[1][1], "x:", neutral_points[0][1])
        #print("Iy value:", Iy[neutral_points[1][1]][neutral_points[0][1]], "of y:", neutral_points[1][1], "x:", neutral_points[0][1])

        print("Proof of neutral point. These points are well below threshold:")
        print("Ix value:", Ix[neutral_points[0][2]][neutral_points[1][2]], "of y:", neutral_points[0][2], "x:", neutral_points[1][2])
        print("Iy value:", Iy[neutral_points[0][2]][neutral_points[1][2]], "of y:", neutral_points[0][2], "x:", neutral_points[1][1])

        #print("If this coordset was switched. (x first) Where prints y first.")
        #print("Ix value:", Ix[neutral_points[1][2]][neutral_points[0][2]])
        #print("Iy value:", Iy[neutral_points[1][2]][neutral_points[0][2]])
    
    # Set locations in image_copy where Ix and Iy are effectively zero to 0
    false_colour[neutral_points] = 0
    
    # Set locations where only Ix is effectively zero to 10
    only_Ix_zero = np.where((np.abs(Ix) <= Ix_threshold) & (np.abs(Iy) > Iy_threshold))
    false_colour[only_Ix_zero] = 10
    
    # Set locations where only Iy is effectively zero to 100
    only_Iy_zero = np.where((np.abs(Ix) > Ix_threshold) & (np.abs(Iy) <= Iy_threshold))
    false_colour[only_Iy_zero] = 100

    if (verboseH):
        print("Sample:")
        print("\nOnly Ix is less than 0:", Ix[only_Ix_zero[0][0]][only_Ix_zero[1][0]], "<", Ix_threshold, "<", Iy[only_Ix_zero[0][0]][only_Ix_zero[1][0]])
        print("Only Iy is less than 0:", Iy[only_Iy_zero[0][0]][only_Iy_zero[1][0]], "<", Iy_threshold, "<",  Ix[only_Iy_zero[0][0]][only_Iy_zero[1][0]])
        #Implement for loop
        
    
    # Return image copy (false colour image) and coordinates of neutral points
    # Old. 0 first then 1.
    # return image_copy, list(zip(neutral_points[0], neutral_points[1]))#neutral_points[0]:ys, neutral_points[1]:xs

    # We could package xs first.
    return false_colour, list(zip(neutral_points[0], neutral_points[1]))

#Discrete Colorbar
def process_determinant(image, strkey, search='', verboseH=False, verboseL=False):
    """
    Takes in an image and finds where is it either > than 0 or less than 0.
    """
    height, width = image.shape

    if search == 'min':
        retarr = np.where(image > 0)
        if(verboseH):
            print("Looking for "+strkey+" coordinates greater than 0")
    elif search == 'max':
        retarr = np.where(image < 0)
        if (verboseH):
            print("Looking for "+strkey+" coordinates less than 0")    
            
    if(verboseH):  
        print("N coordintates discovered:", len(list(zip(retarr[0], retarr[1]))))
        print("")

    return list(zip(retarr[0], retarr[1]))
    
def makeLims(z, std, mean):
    return((z*std)+mean)

def coordsetintersect(coords1, coords2):
    seta=set(coords1)
    setb=set(coords2)

    intersect = seta.intersection(setb)

    return list(intersect)

def present_search(a_dict, verboseH=False):
    """

    """
    
    if (verboseH):
        print("N neutral points:          ", 
              len(a_dict['NeutralPoints']))
        print("Determinants > 0:          ", 
              len(a_dict['Dets>0']))
        print("Intersect neutrals & dets: ", 
              len(coordsetintersect(a_dict['NeutralPoints'], 
              a_dict['Dets>0'])))
    
    localmaxima = coordsetintersect(a_dict['Ixx<0'], a_dict['Dets>0']) #
    neutralmaximus = coordsetintersect(a_dict['NeutralPoints'], localmaxima)
    
    if (verboseH):
        print("N local max     :          ", len(  localmaxima    ))
        print("Neutrallocalmax :          ", len(neutralmaximus))
        print("")

    return (localmaxima, neutralmaximus)

def filter_coords(list_a, xlims=(), ylims=(), verboseH=False):
    """
    Takes in a list of coordinates
    """
    listab = []

    if len(xlims) > 0 and len(ylims)>0:
        for coordset in list_a:
            if( coordset[0] > ylims[0] and  coordset[0] < ylims[1] and coordset[1] > xlims[0] and coordset[1] < xlims[1]):
                listab.append(coordset)
    else:
        raise ValueError('Non acceptable xlims and ylims')
    if verboseH:
        print("Original coordset:", len(list_a))
        print("Filtered coordset:", len(listab))
    return(listab)

def FFThings(inmage):    
    fimage = np.fft.fft2(inmage)
    #print("1. FFT the image", type(fimage), fimage.shape)
    
    absfimage = np.abs(fimage)
    logabsfimage = np.log10(np.abs(fimage))

    shiftfimage = np.fft.fftshift(fimage)
    shiftlogabsfimage = np.fft.fftshift(logabsfimage)   
    shiftabsfimage = np.fft.fftshift(absfimage)   

    return({'fimage': fimage, 
            'fftshiftfimage': shiftfimage, 
            'absfimage':absfimage, 
            'logabsfimage':logabsfimage, 
            'fftshiftlogabsfimage':shiftlogabsfimage,
            'fftshiftabsfimage':shiftabsfimage})

def center_complex_image(image, new_size):
    """
    #This takes in complex valued 
    #np,zeroes(size, dtype=complx)
    """
    newarray = np.zeros(new_size, dtype=complex)
    
    # Extract dimensions of the new image
    new_width, new_height = new_size
        
    # Calculate starting indices to paste the original image
    start_y = (new_height - image.shape[0]) // 2
    start_x = (new_width - image.shape[1]) // 2
    
    # Calculate the end indices
    end_y = start_y + image.shape[0]
    end_x = start_x + image.shape[1]
    
    # Paste the original image into the center of the new image
    newarray[start_y:end_y, start_x:end_x] = image
    
    return newarray

def zero_outside_radius(img, center, radius):
    """
    Needs to work with complex image
    """
    
    # Create an array for the output image, initially filled with zeros
    imx = img.copy()
    height, width = imx.shape
    output_image = np.zeros_like(imx)
    
    # Extract the center coordinates
    cx, cy = center
    
    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate the squared distance from the center
            if ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2:
                # Copy the pixel value if inside the radius
                output_image[y, x] = imx[y, x]
                
    return output_image

def Combine(indict, jointup, verboseC=False):
    #THIS IS WRONG. THIS IS A BUG!
    if (verboseC):
        print("These should be the same")
        print("Last of reversed:", np.array([x[0] for x in indict[jointup[0]]])[::-1][-1])
        print("First of right side:", np.array([x[0] for x in indict[jointup[1]]])[::][0])
    
    joined = np.concatenate((np.array([x[0] for x in indict[jointup[0]]])[::-1][:-1], #Flipped and trimmed. Why do you trim it like that? [:-1]
                             np.array([x[0] for x in indict[jointup[1]]])[::][:]), axis = 0) #[:-1] #Everything until the last one...

    if verboseC:
        print("Vals at start (verify): ", [indict[jointup[0]][0], indict[jointup[1]][0]] ) #This is already proof...
        print("Joining: ",  jointup[0], " with ",  jointup[1]  )
        print("Length of " + jointup[0] + " reversed and pruned: ", len( np.array([x[0] for x in indict[jointup[0]]])[::-1][1:])    )
        print("Length of " + jointup[1] + ": ", len( np.array([x[0] for x in indict[jointup[1]]])[::][:])   )
        print("Final Length:",
              len( np.array([x[0] for x in indict[jointup[0]]])[::-1][:-1]) + len( np.array([x[0] for x in indict[jointup[1]]])[::][:])    )
        
        print("")

        print("-----------------------")
    
    return([joined, 
            len(np.array([x[0] for x in indict[jointup[0]]])[::-1][1:])])

def sample_in_directions(image, cent_loc, verboseH=False, verboseL=False):
    directions = {
        'up': (-1, 0),
        'down': (1, 0),
        'left': (0, -1),
        'right': (0, 1),
        'top_left_diagonal': (-1, -1),
        'top_right_diagonal': (-1, 1),
        'bottom_left_diagonal': (1, -1),
        'bottom_right_diagonal': (1, 1)
    }
    
    img_height, img_width = image.shape
    central_y, central_x = cent_loc
    samples = {}

    if(verboseH):
        print("Amplitude at sampled center (Cartesian) (", cent_loc[1], cent_loc[0],")", image[cent_loc[0]][cent_loc[1]])
        print()
    
    # For each direction, sample pixels and record their positions
    for direction, (dy, dx) in directions.items():
        y, x = central_y, central_x
        samples[direction] = []
        #print()
        
        while 0 <= y < img_height and 0 <= x < img_width:
            # Append both the pixel value and the coordinates as a tuple
            samples[direction].append([image[y, x], (y, x)])
            y += dy
            x += dx

        if (verboseL):
            print("Direction:", direction, "Length sample:", len(samples[direction]))
            print("Val (amp and coord) at start:", samples[direction][0])
            print("Val (amp and coord) at end:", samples[direction][-1])
            print("Height:", samples[direction][0][0] - samples[direction][-1][0])
            print()    
    
    return samples

def rectify_walk(walkdata, lens, joinslist, verboseH = False , verboseL=False):
    #if (verboseH):
    #    print("This illustrates the problem where Index of the data are not the same all the time")
    #    print("Depending on where the peak is")
        
    legendhold = []
    datalens = [] 
    
    for k in range(0, len(walkdata)):# in datasjoined:
        if (verboseL):
            print(joinslist[k][0]+"_to_"+ joinslist[k][1] )
            print("Length of data:" , len(walkdata[k]))
            print("Length flipped: ", lens[k])
            print("") 
            print("Anchor verify:", walkdata[k][lens[k]], "index:", lens[k]) #Lat value... 
            print()
            print("Max of data:", np.max(walkdata[k]), "index:", np.where(walkdata[k]== np.max(walkdata[k])))
            print("")
            print("----")
            print("")
            
        datalens.append(lens[k])
        legendhold.append(joinslist[k][0]+"_to_"+ joinslist[k][1])

    if (verboseH):
        print("Max length (padded must defer to this):", np.max(datalens))
        print("Min length (farthest offset):", np.min(datalens))

        print("\nOffset Corrections:")#, [(np.max(datalens) - x) for x in datalens] )
        #print("Joinslist:", legendhold)
        for k in range(0, len(legendhold)):
            print(legendhold[k],":", [(np.max(datalens) - x) for x in datalens][k], "spaces")
            
    return(legendhold) #We get the corrected model...
    
def mask_image(image, coordinates, mask_val):
    imgcpy = image.copy()
    for coord in coordinates:
        imgcpy[coord]=mask_val
    return imgcpy