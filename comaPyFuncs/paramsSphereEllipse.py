import numpy as np

"""
These functions some stuff
"""

# Define model parameters
normalCircles = 0.1 * (np.arange(10, 320, 10))  # For 100x100
upscaledCircles = np.arange(10, 320, 10)  # For 1000x1000

upscaledEllipses = [
    [15, 10, 65],
    [15, 10, 65],
    [16, 10, 65],
    [20, 10, 65],
    [30, 20, 65],
    [40, 30, 65],
    [50, 40, 65],
    [60, 50, 65],
    [80, 70, 65],
    [100, 90, 65],
    [120, 110, 65],
    [140, 130, 65],
    [150, 143, 65],
    [160, 150, 65],
    [170, 159, 65],
    [180, 167, 65],
    [190, 175, 65],
    [200, 185, 65],
    [210, 190, 65],
    [220, 195, 65],
    [230, 200, 65],
    [240, 205, 65],
    [250, 211, 65],
    [260, 215, 65],
    [270, 223, 65],
    [280, 230, 65],
    [290, 236, 65],
    [300, 244, 65],
    [310, 251, 65],
]
normalEllipses = [
    [int(0.1 * x[0]), int(0.1 * x[1]), x[2]] for x in upscaledEllipses
]  # For 100 x 100
