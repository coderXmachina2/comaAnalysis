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

from scipy.signal import correlate2d
from skimage import io, color

"""
These functions do statistical analysis.
"""

#autocovariance_2d
#img_diag

def hessianStats(image):
    """
    Returns statistics
    """

    Ix = ndimage.sobel(image, axis=0, mode='constant', cval=0.0)  # This is a first derivative
    Iy = ndimage.sobel(image, axis=1, mode='constant', cval=0.0)  # This is a first derivative
    
    Ixx = ndimage.sobel(Ix, axis=0, mode='constant', cval=0.0) 
    Iyy = ndimage.sobel(Iy, axis=1, mode='constant', cval=0.0) 
    
    Ixy = ndimage.sobel(Ix, axis=1, mode='constant', cval=0.0) 
    Iyx = ndimage.sobel(Iy, axis=0, mode='constant', cval=0.0)  # Cross-derivative... The only true second order
    
    # Stack the partial derivatives to form the Hessian matrix at each pixel
    FirstOrder = np.array([Ix, Iy])
    #SecondOrder = np.array([Ixx, Iyy])
    
    Hessian = np.array([[Ixx, Ixy], 
                        [Iyx, Iyy]])
    
    #Calculate determinant
    detHessian = Ixx * Iyy - Ixy * Iyx
    trace_Hessian = Ixx + Iyy
    
    eigenvals, eigenvects = np.linalg.eig(  Hessian  )
    
    return({"FirstOrder": FirstOrder,
            "Hessian": Hessian,
            "DeterminantHessian": detHessian,
            "TraceHessian":  trace_Hessian,
            "EigenValues": eigenvals,
            "EigenVectors": eigenvects}) 