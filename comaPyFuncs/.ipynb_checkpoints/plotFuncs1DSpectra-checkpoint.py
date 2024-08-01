import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy import stats
from statsmodels import robust

"""
These functions return nothing. They simply take in and plot.
"""
def plotCompare2Lines(listData, 
                      listLegend,
                      labels=[], 
                      suptitle='',
                      noise=False):
    """
    Takes in 1D data and plots them. Over engineered for what it does. Find out how to use in non over engineered mode
    """

    if isinstance(listData[0], tuple):
        scatter_major = plt.scatter(listData[0][0],
                                    listData[0][1], label=listLegend[0])
        line_major = plt.plot( listData[0][0]   ,  listData[0][1]     )
    else:
        scatter_major = plt.scatter(np.arange(0, len(listData[0])),
                                listData[0], label=listLegend[0])
        line_major = plt.plot(np.arange(0, len(listData[0])), 
                              listData[0])

    #If the first is tuple
    if isinstance(listData[1], tuple):
        scatter_minor = plt.scatter(listData[1][0],
                                    listData[1][1], label=listLegend[1])
        line_minor = plt.plot(listData[1][0],  listData[1][1])
    else:
        # Plot the semi-minor axis
        scatter_minor = plt.scatter(np.arange(0,len(listData[1])), 
                                    listData[1], label=listLegend[1])
        line_minor = plt.plot(np.arange(0,len(listData[1])), listData[1])
    
    # Create custom legend handles
    custom_major = Line2D([0], [0], color='#1f77b4', marker='o', linestyle='-', markersize=8)
    custom_minor = Line2D([0], [0], color='#ff7f0e', marker='o', linestyle='-', markersize=8)
    
    if noise:
        noise_plt = plt.plot(listData[-1], label='Bkg noise')
        custom_noise = Line2D([0], [0], color='#2ca02c', marker='', linestyle='-', markersize=8)
    plt.title(suptitle)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.grid(alpha=0.2)
    # Create a combined legend handle for semi-major and semi-minor axes
    if noise:
        plt.legend([custom_major, custom_minor, custom_noise], [listLegend[0], listLegend[1], listLegend[-1]])
    else:
        plt.legend([custom_major, custom_minor ], [listLegend[0], listLegend[1]])        
    plt.show()

#Here will be a function. Two plots side by side.
def plot_2subp1D_spectra(spectra, subtitles, suptitle, labels=[], lesiterables=False):
    """
    Plots 2 spectra side by side.
    """
    # Create a 4x4 grid of subplots
    #fig, axes = plt.subplots(1, 2, figsize=(15, 15))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Plot each spectrum in the specified subplot
    if (lesiterables):
        for i in range(0, len(spectra[0])):
            if highlightzero:
                if i==0:
                    axes[0, 0].plot(spectra[0][i], alpha=1, label= subtitles[0][i])
                else:
                    axes[0, 0].plot(spectra[0][i], alpha=0.425, label= subtitles[0][i])
            else:
                    axes[0, 0].plot(spectra[0][i], alpha=0.9, label= subtitles[0][i])
                
        for k in range(0, len(spectra[1])): 
            if highlightzero:
                if k==0:
                    axes[0, 1].plot(spectra[1][k], alpha=1, label= subtitles[1][k])
                else:
                    axes[0, 1].plot(spectra[1][k], alpha=0.425, label= subtitles[1][k])
            else:
                    axes[0, 1].plot(spectra[1][k], alpha=0.9, label= subtitles[1][k])

    else:
        ax1.plot(spectra[0], label= subtitles[0])
        ax2.plot(spectra[1], label= subtitles[1])
        #print(axes[0,0])
    # Adding labels for clarity
    ax1.set_title(subtitles[0])
    ax2.set_title(subtitles[1])

    # Add legends to each plot
    ax1.legend()
    ax2.legend()

    ax1.grid(0.2)
    ax2.grid(0.2)

    fig.suptitle(suptitle, 
                 fontsize=16)#, fontweight='bold')

    # Adjust the layout to ensure subplots fit nicely and avoid overlap
    # plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show the plot
    #plt.show()
    plt.tight_layout()
    plt.show()

def plot_4subp1D_spectra(spectra, subtitles, suptitle, highlightzero=False,lesiterables=False):
    """
    Takes in a list of 4 1D spectra. Plots a 4x4 grid of spectra.
    :param spectra: List of 4 spectra, each a 1D list of data points.
    """
    if len(spectra) != 4:
        raise ValueError("Expected a list of exactly 4 spectra.")

    # Create a 4x4 grid of subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    # Plot each spectrum in the specified subplot
    if (lesiterables):
        for i in range(0, len(spectra[0])):
            if highlightzero:
                if i==0:
                    axes[0, 0].plot(spectra[0][i], alpha=1, label= subtitles[0][i])
                else:
                    axes[0, 0].plot(spectra[0][i], alpha=0.425, label= subtitles[0][i])
            else:
                    axes[0, 0].plot(spectra[0][i], alpha=0.9, label= subtitles[0][i])
                
        for k in range(0, len(spectra[1])): 
            if highlightzero:
                if k==0:
                    axes[0, 1].plot(spectra[1][k], alpha=1, label= subtitles[1][k])
                else:
                    axes[0, 1].plot(spectra[1][k], alpha=0.425, label= subtitles[1][k])
            else:
                    axes[0, 1].plot(spectra[1][k], alpha=0.9, label= subtitles[1][k])

        for m in range(0, len(spectra[2])):
            if highlightzero:
                if m==0:
                    axes[1, 0].plot(spectra[2][m], alpha=1, label= subtitles[2][m])
                else:
                    axes[1, 0].plot(spectra[2][m], alpha=0.425, label= subtitles[2][m])
            else:
                    axes[1, 0].plot(spectra[2][m], alpha=0.9, label= subtitles[2][m])

        for n in range(0, len(spectra[3])):
            if highlightzero:
                if n==0:
                    axes[1, 1].plot(spectra[3][n], alpha=1, label= subtitles[3][n])
                else:
                    axes[1, 1].plot(spectra[3][n], alpha=0.425, label= subtitles[3][n])
            else:
                    axes[1, 1].plot(spectra[3][n], alpha=0.9, label= subtitles[3][n])
    else:
        axes[0, 0].plot(spectra[0], label= subtitles[0])
        axes[0, 1].plot(spectra[1], label= subtitles[1])
        axes[1, 0].plot(spectra[2], label= subtitles[2])
        axes[1, 1].plot(spectra[3], label= subtitles[3])

    # Adding labels for clarity
    axes[0, 0].set_title(subtitles[0][-1])
    axes[0, 1].set_title(subtitles[1][-1])
    axes[1, 0].set_title(subtitles[2][-1])
    axes[1, 1].set_title(subtitles[3][-1])

    # Add legends to each plot
    axes[0, 0].legend()
    axes[0, 1].legend()
    axes[1, 0].legend()
    axes[1, 1].legend()

    axes[0, 0].grid(0.2)
    axes[0, 1].grid(0.2)
    axes[1, 0].grid(0.2)
    axes[1, 1].grid(0.2)

    fig.suptitle(suptitle, 
                 fontsize=16)#, fontweight='bold')

    # Adjust the layout to ensure subplots fit nicely and avoid overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # Show the plot
    plt.show()