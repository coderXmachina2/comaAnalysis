import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import importlib

from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from skimage.draw import ellipse_perimeter, circle_perimeter

from comaPyFuncs import procFuncsStats2
from comaPyFuncs import procFuncsZSortPlease
from comaPyFuncs import plotFuncsIm

importlib.reload(procFuncsStats2)
importlib.reload(procFuncsZSortPlease)
importlib.reload(plotFuncsIm)

"""
These functions always return stuff. These functions are related to Radial Models.

The functions here call other functions. If they do not call on subs from others then they do not qualify for being here.
"""

def centfindpipe(inmage, imtype='' ,innerargs=((),()), zin= 1, plotTrue=False, verbosityF=False, verbosityT=False):
    """
    innerargs[0] = xlims
    innerargs[1] = ylims
    """
    #this only does math. We assume it operates with respect to (0, 0)
    img_Hessian = procFuncsStats2.hessianStats(inmage)
    
    upScaledNeutrals = procFuncsZSortPlease.process_derivative(img_Hessian['FirstOrder'], threshF=zin, verboseH=verbosityF)
    
    coordsetDet = procFuncsZSortPlease.process_determinant(img_Hessian['DeterminantHessian'], "Determinant" ,search='min', verboseH=verbosityF)
    coordsetmax = procFuncsZSortPlease.process_determinant(img_Hessian['Hessian'][0][0], "Ixx", search='max', verboseH=verbosityF) #Greater than 0
    
    adict = {"NeutralPoints":upScaledNeutrals[1],
             "Dets>0":coordsetDet,
             "Ixx<0":coordsetmax}
    
    localmax, neutralmaxs = procFuncsZSortPlease.present_search(adict, verboseH=verbosityF) 
    inner = procFuncsZSortPlease.filter_coords(neutralmaxs, xlims = innerargs[0], ylims=innerargs[1], verboseH=verbosityF) 

    if (verbosityT):
        #print(len(innercords))
        print("Y mean:", int(np.mean([x[0] for x in inner])))
        print("X mean:", int(np.mean([x[1] for x in inner])))
        
        print("Cartesian (x, y): (", int(np.mean([x[1] for x in inner])) ,",", int(np.mean([x[0] for x in inner])),")")
        print("Computational [depth][width]: [",  int(np.mean([x[0] for x in inner])) ,"][",  int(np.mean([x[1] for x in inner])) ,"]")
        print("")

    if (plotTrue):
        plotFuncsIm.plot_1_im(image= procFuncsZSortPlease.set_pixels(inmage, inner),#, sub_val=0.00001), 
                              header='', 
                              title=  imtype+ ' Filtered Center. N local max neutral Points discovered: '+ str(len(inner)) ,
                              spheres=([],[]), #This is with respect to centre
                              ellipses=([],[]), 
                              targeter = [(int(np.mean([x[1] for x in inner])), 1000 - int(np.mean([x[0] for x in inner])))], #targeter is a scatter point 
                              contour=True)
    
    return (inner)

def Fsmoothing(inmage, zeroutparams ,newSize, verboseT=False):
    #Smoothing pipe
    #Fourier Transfor
    fimage = procFuncsZSortPlease.FFThings(inmage)['fimage']
    
    #Shift
    fftshiftfimage = procFuncsZSortPlease.FFThings(inmage)['fftshiftfimage']
    
    #Define a threshold in pixel. Anything outside the threshod set to 0
    zerout = procFuncsZSortPlease.zero_outside_radius(procFuncsZSortPlease.FFThings(inmage)['fftshiftfimage'],
                                                      (zeroutparams[0][0], zeroutparams[0][1]), zeroutparams[1]) 
    
    #Embed
    embed = procFuncsZSortPlease.center_complex_image(zerout, newSize)
    
    #InverseShift
    Ishift = np.fft.ifftshift(embed)
    
    #IFFT
    inverseFFT = np.fft.ifft2(Ishift)
    
    #Abs(IFFT)
    absIFFT = np.abs(inverseFFT)
    
    if (verboseT):
        print("absIFFT shape:", absIFFT.shape)

    return (absIFFT)


    
#######################################################################################################################################