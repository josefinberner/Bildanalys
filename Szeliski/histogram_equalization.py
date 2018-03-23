import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt, cm


def main():
    # Exercise 3.6 Histogram equalization

    # Convert the color image to luminance (section 3.1.2)
    # Compute the histogram, the cumulative distribution and the compensation transfer function (sec. 3.1.4)
    # (opt) Try to increase the punch in the image by ensuring that a certain fraction of pixels, 5% are mapped to pure b/w
    # (opt) Limit the local gain f'(I) in the transfer function (more details)
    # Compensate the luminance channel through the lookup table and re-generate the color image using color ratios (2.116)
    # (opt) Color values that are clipped (=saturated) may appear unnatural when remapped. Fix this!


if __name__ == '__main__':
    main()
    plt.show(block=True)