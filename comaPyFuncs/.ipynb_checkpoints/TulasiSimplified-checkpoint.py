#ComaView
import healpy as hp
import numpy as np
import scipy.ndimage as ndimage
import os, sys
import datetime
import drizzlib
import glob
import importlib

from astropy.io import fits as pyfits
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

hpproj = glob.glob('./Processed/outIMG/hp_proj/*')

from ahamidSupFuncs import plotFuncsIm
from ahamidSupFuncs import plotFuncs1DSpectra
from ahamidSupFuncs import procFuncsModRad1
from ahamidSupFuncs import procFuncsStats1
from ahamidSupFuncs import procFuncsStats2
from ahamidSupFuncs import procFuncsRescale
from ahamidSupFuncs import procFuncsMisc
from ahamidSupFuncs import procFuncaPipe
from ahamidSupFuncs import procFuncsZSortPlease

importlib.reload(plotFuncsIm)
importlib.reload(plotFuncs1DSpectra)
importlib.reload(procFuncsModRad1)
importlib.reload(procFuncsStats1)
importlib.reload(procFuncsStats2)
importlib.reload(procFuncsRescale)
importlib.reload(procFuncsMisc)
importlib.reload(procFuncaPipe)
importlib.reload(procFuncsZSortPlease)

#I am able to plot some circular and elliptical models
offset = 50
image=pyfits.open(hpproj[0])[1].data[(350-offset):(350+offset),(350-offset):(350+offset)]
header=pyfits.open(hpproj[0])[1].header

#Inside 

def cleanAndUpscale(image, cutoff, newsize):
    nx,ny=image.size
    nnx,nny=newsize
    
    fimage=np.fft.fftshift(np.fft.fft2(image))

    # Create a large empty image, and embed fimage at its center
    fnewimage=np.zeros(newsize,dtype=complex)
    idx=(nnx-nx)/2
    idy=(nny-ny)/2
    fnewimage[idx:idx+nx,idy:idy+ny]=fimage

    # Identify the citcle of choice
    x=np.arange(-nnx/2,nnx/2)
    y=np.arange(-nny/2,nny/2)
    xx,yy=np.meshgrid(x,y)
    mask=np.where(np.sqrt(xx**2+yy**2) > r)

    # Use the mask above to set values to zero outside the circle of coice.
    import numpy.ma as ma
    fnewiamge_zeroed=ma.array(fnewimage,mask)

    cleanimage=np.real(np.fft.ifft2(np.fft.ifftshift(fnewimage_zeroed.filled(fillvalue=np.complex(0))))
    return cleanimage

r = 5
center=(50,50)

cleanimage=cleanAndUpscale(image)
plt.imshow(cleanimage)
plt.show()


#ComaView
import healpy as hp
import numpy as np
import scipy.ndimage as ndimage
import os, sys
import datetime
import drizzlib
import glob
import importlib

from astropy.io import fits as pyfits
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

hpproj = glob.glob('./Processed/outIMG/hp_proj/*')

from ahamidSupFuncs import plotFuncsIm
from ahamidSupFuncs import plotFuncs1DSpectra
from ahamidSupFuncs import procFuncsModRad1
from ahamidSupFuncs import procFuncsStats1
from ahamidSupFuncs import procFuncsStats2
from ahamidSupFuncs import procFuncsRescale
from ahamidSupFuncs import procFuncsMisc
from ahamidSupFuncs import procFuncaPipe
from ahamidSupFuncs import procFuncsZSortPlease

importlib.reload(plotFuncsIm)
importlib.reload(plotFuncs1DSpectra)
importlib.reload(procFuncsModRad1)
importlib.reload(procFuncsStats1)
importlib.reload(procFuncsStats2)
importlib.reload(procFuncsRescale)
importlib.reload(procFuncsMisc)
importlib.reload(procFuncaPipe)
importlib.reload(procFuncsZSortPlease)

#I am able to plot some circular and elliptical models
offset = 50
image=pyfits.open(hpproj[0])[1].data[(350-offset):(350+offset),(350-offset):(350+offset)]
header=pyfits.open(hpproj[0])[1].header

#Inside 

def cleanAndUpscale(image,cutoff,newsize):
    fimage=np.fft.fftshift(np.fft.fft2(image))

    height, width = image.shape
    output_image = np.zeros_like(fimage)
    
    # Extract the center coordinates
    cx, cy = fimage.shape
    
    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate the squared distance from the center
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                # Copy the pixel value if inside the radius
                output_image[y, x] = image[y, x]
    
    newarray = np.zeros(newsize, dtype=complex)
    
    # Extract dimensions of the new image
    #workimage=image.copy() 
    new_width, new_height = new_size
    
    # Create a new array of zeros with the new dimensions
    #new_image = np.zeros((new_height, new_width), dtype=workimage.dtype)
    
    # Calculate starting indices to paste the original image
    start_y = (new_height - image.shape[0]) // 2
    start_x = (new_width - image.shape[1]) // 2
    
    # Calculate the end indices
    end_y = start_y + image.shape[0]
    end_x = start_x + image.shape[1]
    
    # Paste the original image into the center of the new image
    newarray[start_y:end_y, start_x:end_x] = image

    cleanimage=np.real(np.fft.ifft2(np.fft.ifftshift(newarray)))
    return newarray

r = 5
center=(50,50)

cleanimage=cleanAndUpscale(image,r,(500,500))
plt.imshow(cleanimage)
plt.show()

nnx=nny=500
r=22
x=np.arange(-nnx/2,nnx/2)
y=np.arange(-nny/2,nny/2)
xx,yy=np.meshgrid(x,y)
mask=np.where(np.sqrt(xx**2+yy**2) < r)

#ComaView
import healpy as hp
import numpy as np
import scipy.ndimage as ndimage
import os, sys
import datetime
import drizzlib
import glob
import importlib

from astropy.io import fits as pyfits
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

hpproj = glob.glob('./Processed/outIMG/hp_proj/*')

from ahamidSupFuncs import plotFuncsIm
from ahamidSupFuncs import plotFuncs1DSpectra
from ahamidSupFuncs import procFuncsModRad1
from ahamidSupFuncs import procFuncsStats1
from ahamidSupFuncs import procFuncsStats2
from ahamidSupFuncs import procFuncsRescale
from ahamidSupFuncs import procFuncsMisc
from ahamidSupFuncs import procFuncaPipe
from ahamidSupFuncs import procFuncsZSortPlease

importlib.reload(plotFuncsIm)
importlib.reload(plotFuncs1DSpectra)
importlib.reload(procFuncsModRad1)
importlib.reload(procFuncsStats1)
importlib.reload(procFuncsStats2)
importlib.reload(procFuncsRescale)
importlib.reload(procFuncsMisc)
importlib.reload(procFuncaPipe)
importlib.reload(procFuncsZSortPlease)

#I am able to plot some circular and elliptical models
offset = 50
image=pyfits.open(hpproj[0])[1].data[(350-offset):(350+offset),(350-offset):(350+offset)]
header=pyfits.open(hpproj[0])[1].header

#Inside 

def cleanAndUpscale(image,cutoff,newsize):
    nx,ny=image.size
    nnx,nny=newsize
    
    fimage=np.fft.fftshift(np.fft.fft2(image))

    # Create a large empty image, and embed fimage at its center
    fnewimage=np.zeros(newsize,dtype=complex)
    idx=(nnx-nx)/2
    idy=(nny-ny)/2
    fnewimage[idx:idx+nx,idy:idy+ny]=fimage

    # Identify the citcle of choice
    x=np.arange(-nnx/2,nnx/2)
    y=np.arange(-nny/2,nny/2)
    xx,yy=np.meshgrid(x,y)
    mask=np.where(np.sqrt(xx**2+yy**2) > cutoff)

    # Use the mask above to set values to zero outside the circle of coice.
    import numpy.ma as ma
    fnewiamge_zeroed=ma.array(fnewimage,mask)

    cleanimage=np.real(np.fft.ifft2(np.fft.ifftshift(fnewimage_zeroed.filled(fillvalue=np.complex(0)))))
    return cleanimage

r = 5
center=(50,50)

cleanimage=cleanAndUpscale(image,r,(500,500))
plt.imshow(cleanimage)
plt.show()

#ComaView
import healpy as hp
import numpy as np
import scipy.ndimage as ndimage
import os, sys
import datetime
import drizzlib
import glob
import importlib

from astropy.io import fits as pyfits
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

hpproj = glob.glob('./Processed/outIMG/hp_proj/*')

from ahamidSupFuncs import plotFuncsIm
from ahamidSupFuncs import plotFuncs1DSpectra
from ahamidSupFuncs import procFuncsModRad1
from ahamidSupFuncs import procFuncsStats1
from ahamidSupFuncs import procFuncsStats2
from ahamidSupFuncs import procFuncsRescale
from ahamidSupFuncs import procFuncsMisc
from ahamidSupFuncs import procFuncaPipe
from ahamidSupFuncs import procFuncsZSortPlease

importlib.reload(plotFuncsIm)
importlib.reload(plotFuncs1DSpectra)
importlib.reload(procFuncsModRad1)
importlib.reload(procFuncsStats1)
importlib.reload(procFuncsStats2)
importlib.reload(procFuncsRescale)
importlib.reload(procFuncsMisc)
importlib.reload(procFuncaPipe)
importlib.reload(procFuncsZSortPlease)

#I am able to plot some circular and elliptical models
offset = 50
image=pyfits.open(hpproj[0])[1].data[(350-offset):(350+offset),(350-offset):(350+offset)]
header=pyfits.open(hpproj[0])[1].header

#Inside 

def cleanAndUpscale(image,cutoff,newsize):
    fimage=np.fft.fftshift(np.fft.fft2(image))

    height, width = image.shape
    output_image = np.zeros_like(fimage)
    
    # Extract the center coordinates
    cx, cy = fimage.shape
    
    # Iterate over each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate the squared distance from the center
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                # Copy the pixel value if inside the radius
                output_image[y, x] = image[y, x]
    
    newarray = np.zeros(newsize, dtype=complex)
    
    # Extract dimensions of the new image
    #workimage=image.copy() 
    new_width, new_height = new_size
    
    # Create a new array of zeros with the new dimensions
    #new_image = np.zeros((new_height, new_width), dtype=workimage.dtype)
    
    # Calculate starting indices to paste the original image
    start_y = (new_height - image.shape[0]) // 2
    start_x = (new_width - image.shape[1]) // 2
    
    # Calculate the end indices
    end_y = start_y + image.shape[0]
    end_x = start_x + image.shape[1]
    
    # Paste the original image into the center of the new image
    newarray[start_y:end_y, start_x:end_x] = image

    cleanimage=np.real(np.fft.ifft2(np.fft.ifftshift(newarray)))
    return newarray

r = 5
center=(50,50)

cleanimage=cleanAndUpscale(image,r,(500,500))
plt.imshow(cleanimage)
plt.show()

nnx=nny=500
r=22
x=np.arange(-nnx/2,nnx/2)
y=np.arange(-nny/2,nny/2)
xx,yy=np.meshgrid(x,y)
mask=np.where(np.sqrt(xx**2+yy**2) < r)

#ComaView
import healpy as hp
import numpy as np
import scipy.ndimage as ndimage
import os, sys
import datetime
import drizzlib
import glob
import importlib

from astropy.io import fits as pyfits
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse

hpproj = glob.glob('./Processed/outIMG/hp_proj/*')

from ahamidSupFuncs import plotFuncsIm
from ahamidSupFuncs import plotFuncs1DSpectra
from ahamidSupFuncs import procFuncsModRad1
from ahamidSupFuncs import procFuncsStats1
from ahamidSupFuncs import procFuncsStats2
from ahamidSupFuncs import procFuncsRescale
from ahamidSupFuncs import procFuncsMisc
from ahamidSupFuncs import procFuncaPipe
from ahamidSupFuncs import procFuncsZSortPlease

importlib.reload(plotFuncsIm)
importlib.reload(plotFuncs1DSpectra)
importlib.reload(procFuncsModRad1)
importlib.reload(procFuncsStats1)
importlib.reload(procFuncsStats2)
importlib.reload(procFuncsRescale)
importlib.reload(procFuncsMisc)
importlib.reload(procFuncaPipe)
importlib.reload(procFuncsZSortPlease)

#I am able to plot some circular and elliptical models
offset = 50
image=pyfits.open(hpproj[0])[1].data[(350-offset):(350+offset),(350-offset):(350+offset)]
header=pyfits.open(hpproj[0])[1].header

#Inside 

def cleanAndUpscale(image,cutoff,newsize):
    nx,ny=image.size
    nnx,nny=newsize
    
    fimage=np.fft.fftshift(np.fft.fft2(image))

    # Create a large empty image, and embed fimage at its center
    fnewimage=np.zeros(newsize,dtype=complex)
    idx=(nnx-nx)/2
    idy=(nny-ny)/2
    fnewimage[idx:idx+nx,idy:idy+ny]=fimage

    # Identify the citcle of choice
    x=np.arange(-nnx/2,nnx/2)
    y=np.arange(-nny/2,nny/2)
    xx,yy=np.meshgrid(x,y)
    mask=np.where(np.sqrt(xx**2+yy**2) > cutoff)

    # Use the mask above to set values to zero outside the circle of coice.
    import numpy.ma as ma
    fnewiamge_zeroed=ma.array(fnewimage,mask)

    cleanimage=np.real(np.fft.ifft2(np.fft.ifftshift(fnewimage_zeroed.filled(fillvalue=np.complex(0)))))
    return cleanimage

r = 5
center=(50,50)

cleanimage=cleanAndUpscale(image,r,(500,500))
plt.imshow(cleanimage)
plt.show()



################################################################

fig, ax = plt.subplots(figsize=(16, 12))
# Generate contour levels
levels = np.linspace(np.min(upscaled_img), 
                     np.max(upscaled_img), 
                     num=7)
        
# Create contour lines over the image
contours = ax.contour(upscaled_img, levels,
                      colors='red', origin='lower', alpha=0.3,
                      extent=[0, upscaled_img.shape[1], 0, upscaled_img.shape[0]])
plt.show()