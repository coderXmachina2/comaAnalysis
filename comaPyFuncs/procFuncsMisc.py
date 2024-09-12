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

"""
These functions always return stuff. Ideas that were not confirmed.
"""

# convert_to_equatorial
# sliding_difference
# sliding_difference_no_wrap


def is_within_ellipse(x, y, center, a, b, theta):
    # print("Checking is within ellipse!")
    h, k = center
    cos_angle = np.cos(theta)
    sin_angle = np.sin(theta)

    x_shifted = x - h
    y_shifted = y - k

    x_rotated = x_shifted * cos_angle + y_shifted * sin_angle
    y_rotated = -x_shifted * sin_angle + y_shifted * cos_angle

    ellipse_eq = (x_rotated**2) / (a**2) + (y_rotated**2) / (b**2)

    return ellipse_eq <= 1


def convert_to_equatorial(header):
    """
    Ensures that the WCS header is set to equatorial coordinates (RA/Dec).
    Converts galactic to equatorial if necessary.

    Args:
    header (FITS header): The original FITS header

    Returns:
    WCS header modified for RA/Dec if necessary
    """
    wcs_obj = WCS(header)
    if wcs_obj.wcs.lngtyp != "RA":
        # Assuming the original is in galactic coordinates and needs conversion
        wcs_obj.wcs.set()
        wcs_obj.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs_obj.to_header()


def sliding_difference(data1, d1blurred, max_depth):  # coma y map  # smooth model
    """
    Not saying this is useless but hold on to it. It may come in handy later
    This one wraps around. Edges go to the other side.
    Does not compute the difference at the center

    Return:
    """
    differences = []
    center = np.array(data1.shape) // 2

    for L in range(1, max_depth + 1):
        for dx in range(-L, L + 1):
            for dy in range(-L, L + 1):
                if (
                    np.abs(dx) == L or np.abs(dy) == L
                ):  # Check if at the boundary of depth L
                    shifted_d1blurred = np.roll(d1blurred, shift=(dx, dy), axis=(0, 1))

                    # Calculate difference only where there is overlap
                    overlap_diff = data1 - shifted_d1blurred
                    differences.append(overlap_diff)

    return differences


def sliding_difference_no_wrap(data1, d1blurred, max_depth):
    """
    Not saying this is useless but hold on to it. It may come in handy later
    This one leaves 0s/ blank edges...
    Does not compute the difference at the center

    Return:
    """
    differences = []
    height, width = data1.shape

    for L in range(1, max_depth + 1):
        for dx in range(-L, L + 1):
            for dy in range(-L, L + 1):
                if np.abs(dx) == L or np.abs(dy) == L:
                    shifted_d1blurred = np.zeros_like(d1blurred)

                    # Calculate valid slice ranges
                    x_slice = slice(max(0, dx), min(width, width + dx))
                    y_slice = slice(max(0, dy), min(height, height + dy))

                    # Place shifted image in the correct position
                    shifted_d1blurred[
                        max(0, -dx) : min(width, width - dx),
                        max(0, -dy) : min(height, height - dy),
                    ] = d1blurred[x_slice, y_slice]

                    # Calculate difference only where there is overlap
                    overlap_diff = data1 - shifted_d1blurred
                    differences.append(overlap_diff)

    return np.array(differences)
