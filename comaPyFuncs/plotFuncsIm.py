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
These functions simply take in data and plot 2D images. These functions return nothing.

#1. plot_1_im():
#2. plot_2_im_row():
#3. plot_3_im_row():
#4. plot_img_and_diag4x4()
"""

masterAlpha = 0.2
contourAlpha = 0.2
cenAlpha = 0.2


def plot_1_im(
    image="",
    header="",
    title="",
    spheres=[(), ()],
    ellipses=[(), ()],
    targeter=[()],
    contour=False,
):
    """
    Plots single image with title, colorbars for each image, and ensures that the coordinates are in RA and Dec (equatorial coordinates) if
    wcs header is supplied

    Args:
    image (np.ndarray): List containing three image arrays.
    wcshead (FITS header): Single WCS header.
    title (list of str): List containing three strings for the subtitles of the images.
    spheres (list of integers):
    ellipses (list of lists):
    contour (bool): Attempts coordinate conversion.

    Returns
    N/A
    """
    if header:
        wcs = WCS(header)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(16, 12), subplot_kw={"projection": wcs})
    else:
        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(16, 12))

    # Display the image
    im = ax.imshow(
        image,
        cmap="viridis",
        origin="upper",  # Is the true origin. I asked for upper
        extent=[0, image.shape[1], 0, image.shape[0]],
        vmin=np.min(image),
        vmax=np.max(image),
    )

    # Add a colorbar to the image
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Intensity")

    if contour:
        # Generate contour levels
        levels = np.linspace(np.min(image), np.max(image), num=7)

        contours = ax.contour(
            image,
            levels,
            colors="red",
            origin="upper",
            alpha=contourAlpha,
            extent=[0, image.shape[1], 0, image.shape[0]],
        )

        # Add labels to the contours
        ax.clabel(contours, inline=True, fontsize=8, fmt="%.2e")
        #
    if len(spheres[1]) > 0:
        for idx, rads in enumerate(spheres[1]):
            centex, centey = (
                int(image.shape[0] / 2) - spheres[0][0],
                int(image.shape[1] / 2) - spheres[0][1],
            )

            circout = Circle(
                (centex, centey),
                int(rads),
                color="red",
                fill=False,
                linestyle="--",
                linewidth=1.5,
                alpha=masterAlpha,
            )

            ax.add_patch(circout)

            # Only add horizontal and vertical axes for the last circle
            if idx == len(spheres[1]) - 1:
                ax.plot(
                    [centex - rads, centex + rads],
                    [centey, centey],
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=masterAlpha,
                )
                ax.plot(
                    [centex, centex],
                    [centey - rads, centey + rads],
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=masterAlpha,
                )

    if len(ellipses[1]) > 0:
        for ellist in ellipses[1]:
            centex, centey = (image.shape[0] / 2) - ellipses[0][0], (
                image.shape[1] / 2
            ) - ellipses[0][1]

            semi_major_axis = ellist[0]
            semi_minor_axis = ellist[1]
            incangle = ellist[2]

            ellipse = Ellipse(
                (centex, centey),
                2 * semi_major_axis,
                2 * semi_minor_axis,
                angle=incangle,
                edgecolor="red",
                fill=False,
                facecolor="none",
                linestyle="--",
                linewidth=1.5,
                alpha=masterAlpha,
            )

            # Plot the semi-major axis... If it is the last one. Plot the axes.
            if ellist[0] == ellipses[1][-1][0]:
                # Calculate endpoints of the semi-major and semi-minor axes
                end_major_x = semi_major_axis * np.cos(np.deg2rad(incangle))
                end_major_y = semi_major_axis * np.sin(np.deg2rad(incangle))
                end_minor_x = semi_minor_axis * np.cos(np.deg2rad(incangle + 90))
                end_minor_y = semi_minor_axis * np.sin(np.deg2rad(incangle + 90))

                ax.plot(
                    [centex - end_major_x, centex + end_major_x],
                    [centey - end_major_y, centey + end_major_y],
                    color="red",
                    linestyle="-.",
                    linewidth=1.5,
                    alpha=0.16,
                )

                # Plot the semi-minor axis
                ax.plot(
                    [centex - end_minor_x, centex + end_minor_x],
                    [centey - end_minor_y, centey + end_minor_y],
                    color="red",
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.16,
                )

            ax.add_patch(ellipse)

    # Set the title and labels
    ax.axvline(x=image.shape[0] / 2, color="red", alpha=cenAlpha, linestyle="--")
    ax.axhline(y=image.shape[1] / 2, color="red", alpha=cenAlpha, linestyle="--")

    if len(targeter) > 0:
        for i in range(0, len(targeter)):
            # Instead of using this use a 'plot'
            plt.scatter(targeter[i][0], targeter[i][1], marker=(5, 2), color="red")
            plt.plot(
                [image.shape[0] / 2, targeter[i][0]],
                [image.shape[1] / 2, targeter[i][1]],
                linestyle="--",
                color="red",
            )
    ax.set_title(title)

    # Show the plot
    plt.show()


def plot_2_im_row(
    images,
    spheres=[],
    ellipses=[],
    targeter=[],
    subTitles=[],
    superTitles="",
    cmapN="",
    contour=False,
    gridl=False,
):
    """
    Args: Needs ability to plot ellipse
    images (list of np.ndarray): List containing 2 images.
    """
    # Create a figure and subplots
    if gridl:
        masteralpha = 0.22
    else:
        masteralpha = 0

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

    if not cmapN:
        im1 = axes[0].imshow(images[0], origin="upper", cmap="viridis")
    else:
        im1 = axes[0].imshow(images[0], origin="upper", cmap=cmapN)

    axes[0].set_title(subTitles[0])
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)  # Add colorbar

    axes[0].grid(alpha=0.2)
    axes[0].axvline(x=images[0].shape[0] / 2, color="red", alpha=0.2, linestyle="--")
    axes[0].axhline(y=images[0].shape[1] / 2, color="red", alpha=0.2, linestyle="--")

    # Generate contour levels
    # It can do a circle but it cannot plot an ellipse
    if contour:
        levels = np.linspace(np.min(images[0]), np.max(images[0]), num=7)

        # Create contour lines over the image
        contours = axes[0].contour(
            images[0],
            levels,
            colors="red",
            origin="lower",
            alpha=0.32,
            extent=[0, images[0].shape[1], 0, images[0].shape[0]],
        )
        # Add labels to the contours
        axes[0].clabel(contours, inline=True, fontsize=8, fmt="%.2e")

    if len(spheres) > 0:  # If not empty list
        # for rad in spheres[0]: #Left
        centex, centey = (
            spheres[0]["center"][1],
            spheres[0]["center"][0],
        )  # This goes to the ce
        rads = int(spheres[0]["params"][0])

        circout = Circle(
            (centex, centey),
            rads,
            color="red",
            fill=False,
            linestyle="--",
            linewidth=1.5,
            alpha=0.32,
        )

        axes[0].add_patch(circout)

        axes[0].plot(
            [centex - rads, centex + rads],
            [centey, centey],
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=masterAlpha,
        )
        axes[0].plot(
            [centex, centex],
            [centey - rads, centey + rads],
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=masterAlpha,
        )
    """
    if len(ellipses)>0: #If not empty list
        for ellist in ellipses[1]:
            centex, centey = ellipses[1]['center'][1], ellipses[1]['center'][0] #This goes to the ce

            semi_major_axis = ellipses[1]['params'][0]
            semi_minor_axis = ellipses[1]['params'][1]
            incangle = ellipses[1]['params'][2]

            ellipse = Ellipse((centex, centey),
                              2*semi_major_axis, 2*semi_minor_axis,
                              angle=incangle, edgecolor='red',
                              fill=False,
                              facecolor='none', linestyle='--',
                              linewidth=1.5, alpha=masterAlpha)

           # Plot the semi-major axis... If it is the last one. Plot the axes.
            if ellist[0] == ellipses[0][-1][0]:
                # Calculate endpoints of the semi-major and semi-minor axes
                end_major_x = semi_major_axis * np.cos(np.deg2rad(incangle))
                end_major_y = semi_major_axis * np.sin(np.deg2rad(incangle))
                end_minor_x = semi_minor_axis * np.cos(np.deg2rad(incangle + 90))
                end_minor_y = semi_minor_axis * np.sin(np.deg2rad(incangle + 90))


                axes[0].plot([centex - end_major_x, centex + end_major_x],
                        [centey - end_major_y, centey + end_major_y],
                        color='red', linestyle='-.', linewidth=1.5, alpha=0.16)

                # Plot the semi-minor axis
                axes[0].plot([centex - end_minor_x, centex + end_minor_x],
                        [centey - end_minor_y, centey + end_minor_y],
                        color='red', linestyle='--', linewidth=1.5, alpha=0.16)

            axes[0].add_patch(ellipse)
    """
    if not cmapN:
        im2 = axes[1].imshow(images[1], origin="upper", cmap="viridis")  # flippage
    else:
        im2 = axes[1].imshow(images[1], origin="upper", cmap=cmapN)

    axes[1].set_title(subTitles[1])
    axes[1].axvline(x=images[1].shape[0] / 2, color="red", alpha=0.2, linestyle="--")
    axes[1].axhline(y=images[1].shape[0] / 2, color="red", alpha=0.2, linestyle="--")

    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)  # Add colorbar
    axes[1].grid(alpha=0.2)

    # Generate contour levels
    if contour:
        # Generate contour levels
        levels = np.linspace(np.min(images[1]), np.max(images[1]), num=7)

        # Create contour lines over the image
        contours = axes[1].contour(
            images[1],
            levels,
            colors="red",
            origin="lower",
            alpha=0.32,
            extent=[0, images[1].shape[1], 0, images[1].shape[0]],
        )
        # Add labels to the contours
        axes[1].clabel(contours, inline=True, fontsize=8, fmt="%.2e")
    """
    #Surpressed for the center search
    #This is not
    if len(spheres):
        for rad in spheres[1]:
            centex, centey = images[1].shape[0]/2, images[1].shape[1]/2

            circout = Circle((centex, centey),
                             rad, color='red',
                             fill=False,
                             linestyle='--',
                             linewidth=1.5,
                             alpha=0.32)

            axes[1].add_patch(circout)
    """
    if len(ellipses) > 0:  # If not empty list
        # for ellist in ellipses[1]:
        centex, centey = ellipses[1]["center"][1], ellipses[1]["center"][0]

        semi_major_axis = ellipses[1]["params"][0][0]
        semi_minor_axis = ellipses[1]["params"][0][1]
        incangle = ellipses[1]["params"][0][2]

        ellipse = Ellipse(
            (centex, centey),
            2 * semi_major_axis,
            2 * semi_minor_axis,
            angle=incangle,
            edgecolor="red",
            fill=False,
            facecolor="none",
            linestyle="--",
            linewidth=1.5,
            alpha=masterAlpha,
        )

        # Plot the semi-major axis... If it is the last one. Plot the axes.
        # if ellist[0] == ellipses[1][-1][0]:
        # Calculate endpoints of the semi-major and semi-minor axes
        end_major_x = semi_major_axis * np.cos(np.deg2rad(incangle))
        end_major_y = semi_major_axis * np.sin(np.deg2rad(incangle))
        end_minor_x = semi_minor_axis * np.cos(np.deg2rad(incangle + 90))
        end_minor_y = semi_minor_axis * np.sin(np.deg2rad(incangle + 90))

        axes[1].plot(
            [centex - end_major_x, centex + end_major_x],
            [centey - end_major_y, centey + end_major_y],
            color="red",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.16,
        )

        # Plot the semi-minor axis
        axes[1].plot(
            [centex - end_minor_x, centex + end_minor_x],
            [centey - end_minor_y, centey + end_minor_y],
            color="red",
            linestyle="--",
            linewidth=1.5,
            alpha=0.16,
        )

        axes[1].add_patch(ellipse)

    if len(targeter) > 0:
        for i in range(0, len(targeter)):
            if i > 0:
                scatterc = "green"
            else:
                scatterc = "red"

            axes[0].scatter(
                targeter[i][0], targeter[i][1], marker=(5, 2), color=scatterc
            )
            axes[0].plot(
                [images[0].shape[0] / 2, targeter[i][0]],
                [images[0].shape[1] / 2, targeter[i][1]],
                linestyle="--",
                color=scatterc,
            )

            axes[1].scatter(
                targeter[i][0], targeter[i][1], marker=(5, 2), color=scatterc
            )
            axes[1].plot(
                [images[0].shape[0] / 2, targeter[i][0]],
                [images[0].shape[1] / 2, targeter[i][1]],
                linestyle="--",
                color=scatterc,
            )

    fig.suptitle(superTitles)
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_3_im_row(
    images,
    headers=[],
    subTitles=[],
    circrad=0,
    contour=False,
    ellipseparam=[],
    coordsys="",
):
    """
    Plots three science images side by side (in a row) with subtitles, colorbars for each image,
    and ensures that the coordinates are in RA and Dec (equatorial coordinates).

    Args:
    images (list of np.ndarray): List containing three image arrays.
    headers (list of FITS headers): List containing three WCS headers.
    titles (list of str): List containing three strings for the subtitles of the images.
    circad (float): Radius of circle
    ellipseparam (list): List containing parameters that describe an ellipse
    equatorial (bool):

    Returns
    N/A
    """
    if len(images) != 3 or len(subTitles) != 3 or len(headers) != 3:
        raise ValueError(
            "Images, headers, and titles lists must each contain exactly three elements."
        )

    # Create a figure
    fig = plt.figure(figsize=(18, 6))

    if coordsys == "equatorial":
        # Ensure headers are set to RA/Dec if they are not
        WCShead = [convert_to_equatorial(header) for header in headers]
    elif coordsys == "galactic":
        WCShead = headers
    else:
        zx = 0
    # Add subplots with specific WCS projection
    axs = [fig.add_subplot(1, 3, i + 1, projection=WCS(WCShead[i])) for i in range(3)]

    for i, ax in enumerate(axs):
        # Display image with WCS projection in RA and Dec
        im = ax.imshow(
            images[i],
            aspect="auto",
            cmap="viridis",
            norm=Normalize(vmin=np.min(images[i]), vmax=np.max(images[i])),
        )

        # Add a colorbar for each subplot
        cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=1)
        cbar.set_label("Intensity", fontsize=16)
        cbar.ax.tick_params(labelsize=16)

        if circrad != 0:
            center_x, center_y = images[i].shape[1] / 2, images[i].shape[0] / 2
            circle = Circle(
                (center_x, center_y),
                circrad,
                color="red",
                fill=False,
                linestyle="--",
                linewidth=1.5,
                alpha=0.35,
            )
            ax.add_patch(circle)

        if ellipseparam:
            center_x, center_y = images[i].shape[1] / 2, images[i].shape[0] / 2
            # Draw an ellipse at the center of the image
            ellipse = Ellipse(
                (center_x, center_y),
                2 * ellipseparam[0],
                2 * ellipseparam[1],
                angle=ellipseparam[2],
                edgecolor="red",
                fill=False,
                facecolor="none",
                linestyle="--",
                linewidth=1.5,
                alpha=0.7,
            )
            ax.add_patch(ellipse)

        if contour:
            # Generate contour levels
            levels = np.linspace(np.min(images[1]), np.max(images[1]), num=7)

            # Create contour lines over the image
            contours = axes.contour(
                images[1],
                levels,
                colors="red",
                origin="upper",
                alpha=0.32,
                extent=[0, images[1].shape[1], 0, images[1].shape[0]],
            )
            # Add labels to the contours
            axes.clabel(contours, inline=True, fontsize=8, fmt="%.2e")

        ax.set_title(subTitles[i])
        ax.axis("on")  # Ensure axes are on

        # Draw center lines
        ax.axvline(x=images[i].shape[1] / 2, linestyle="--", color="red", alpha=0.35)
        ax.axhline(y=images[i].shape[0] / 2, linestyle="--", color="red", alpha=0.35)

        # Add a grid for major ticks
        ax.grid(
            True,
            which="major",
            color="white",
            linestyle="--",
            linewidth=0.25,
            alpha=0.35,
        )

    plt.tight_layout()
    plt.show()


def plot_img_and_diag4x4(
    image, header, inbins=256, supT="", circrad=0, ellipseparam=[], logim=False
):
    """
    Returns
    N/A
    """
    # Create a WCS object
    # wcs = WCS(header)

    # Create subplots with WCS for specific subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(supT, fontsize=16)

    # Adding axes manually to specify which should have the WCS projection
    ax0 = fig.add_subplot(2, 2, 1)  # , projection=wcs)  # WCS for the first subplot
    ax1 = fig.add_subplot(2, 2, 2)
    ax2 = fig.add_subplot(2, 2, 3)
    ax3 = fig.add_subplot(2, 2, 4)  # , projection=wcs)  # WCS for the fourth subplot

    # Plot raw input image with WCS
    # Lets make gridlines!
    im = ax0.imshow(image, origin="upper", cmap="viridis")
    cbar1 = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    cbar1.ax.tick_params(labelsize=16)
    cbar1.set_label("Intensity", fontsize=16)
    ax0.grid(alpha=0.2)
    ax0.set_xlabel("RA", fontsize=16)
    ax0.set_ylabel("Dec", fontsize=16)
    ax0.set_title("Image", fontsize=16)

    if circrad != 0:
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        circle = Circle(
            (center_x, center_y),
            circrad,
            color="red",
            fill=False,
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
        )
        circpixes = extract_pixels_within_circle(
            image, (int(center_x), int(center_y)), circrad
        )
        ax0.add_patch(circle)

    if ellipseparam:
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        # Draw an ellipse at the center of the image
        ellipse = Ellipse(
            (center_x, center_y),
            2 * ellipseparam[0],
            2 * ellipseparam[1],
            angle=ellipseparam[2],
            edgecolor="red",
            fill=False,
            facecolor="none",
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
        )
        ellipsepixes = extract_pixels_within_ellipse(
            image,
            (int(center_x), int(center_y)),
            ellipseparam[0],
            ellipseparam[1],
            ellipseparam[2],
        )
        ax0.add_patch(ellipse)

    ax0.axhline(y=image.shape[0] / 2, color="red", linestyle="--", alpha=0.25)
    ax0.axvline(x=image.shape[1] / 2, color="red", linestyle="--", alpha=0.25)

    ax0.tick_params(axis="both", which="major", labelsize=16)

    # Plot histogram
    ax1.hist(image.ravel(), bins=inbins, color="gray", alpha=0.7)
    ax1.axvline(x=np.mean(image.ravel()), color="red", linestyle="--", alpha=0.25)
    ax1.axvline(x=np.median(image.ravel()), color="blue", linestyle="--", alpha=0.25)
    ax1.grid()
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.set_title("Histogram", fontsize=16)
    ax1.set_xlabel("Pixel Value", fontsize=16)
    ax1.set_ylabel("Frequency", fontsize=16)

    # Plot raveled 1D image
    ax2.plot(image.ravel(), color="gray", alpha=0.7)
    ax2.text(
        0,
        np.mean(image.ravel()),
        "mean:" + str(np.round(np.mean(image.ravel()), 8)),
        color="red",
        fontsize=16,
    )
    ax2.axhline(y=np.mean(image.ravel()), color="red", linestyle="--", alpha=0.25)
    ax2.axvline(x=len(image) * len(image) / 2, color="red", linestyle="--", alpha=0.25)
    ax2.set_title("Raveled 1D Image", fontsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.grid()
    ax2.set_xlabel("Index", fontsize=16)
    ax2.set_ylabel("Value", fontsize=16)

    # Filter image with 3 sigma
    if logim:
        im = ax3.imshow(np.log(image), cmap="viridis")

        cbar2 = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar2.set_label("Log Intensity", fontsize=16)
        ax3.set_title("Log Image", fontsize=16)
    else:
        mean = np.mean(image)
        std = np.std(image)
        filtered_image = np.clip(image, mean - 3 * std, mean + 3 * std)

        # Plot filtered image with WCS
        im = ax3.imshow(filtered_image, cmap="viridis")

        ax3.set_title("Filtered Image (3 Sigma)", fontsize=16)

        cbar2 = fig.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        cbar2.set_label("Intensity", fontsize=16)
        cbar2.ax.tick_params(labelsize=16)

    if circrad != 0:
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        circle = Circle(
            (center_x, center_y),
            circrad,
            color="red",
            fill=False,
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
        )
        circpixes = extract_pixels_within_circle(
            image, (int(center_x), int(center_y)), circrad
        )
        ax3.add_patch(circle)

    if ellipseparam:
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        # Draw an ellipse at the center of the image
        ellipse = Ellipse(
            (center_x, center_y),
            2 * ellipseparam[0],
            2 * ellipseparam[1],
            angle=ellipseparam[2],
            edgecolor="red",
            fill=False,
            facecolor="none",
            linestyle="--",
            linewidth=1.5,
            alpha=0.55,
        )
        ellipsepixes = extract_pixels_within_ellipse(
            image,
            (int(center_x), int(center_y)),
            ellipseparam[0],
            ellipseparam[1],
            ellipseparam[2],
        )
        ax3.add_patch(ellipse)

    ax3.grid(alpha=0.2)
    ax3.axhline(y=image.shape[0] / 2, color="red", linestyle="--", alpha=0.25)
    ax3.axvline(x=image.shape[1] / 2, color="red", linestyle="--", alpha=0.25)
    ax3.tick_params(axis="both", which="major", labelsize=16)
    ax3.set_xlabel("RA", fontsize=16)
    ax3.set_ylabel("Dec", fontsize=16)

    plt.tight_layout()
    plt.show()


# Display the results
def plot_image_with_border(image, border_pixels, title):
    """
    Not to be disregarded. This can actually take in a coordinate set and plot out what that coordinate set is...
    """
    plt.imshow(image, cmap="viridis")
    y_coords, x_coords = zip(*border_pixels)
    plt.scatter(x_coords, y_coords, c="red", s=1)
    plt.title(title)
    plt.show()
