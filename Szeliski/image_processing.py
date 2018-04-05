import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt, cm
import copy

def plot_result(imgs, titles):
    nbr_imgs = imgs.__len__()
    print(nbr_imgs)
    fig, axes = plt.subplots(1, nbr_imgs, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    for ind in np.arange(0, nbr_imgs):
        ax[ind].imshow(imgs[ind])
        ax[ind].set_title(titles[ind])

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()


def hist_eq(img_lum, img_color):

    def hist_cum_comp(img):
        pix_vals = np.arange(0, 257, 1)
        hist_vec, bins = np.histogram(img, pix_vals)
        nbr_pixels = img.shape[0] * img.shape[1]
        cum = np.zeros([hist_vec.__len__(), 1])
        for ind in pix_vals[0:256]:
            if ind == 0:
                cum[ind] = hist_vec[ind] / nbr_pixels
            else:
                cum[ind] = cum[ind - 1] + hist_vec[ind] / nbr_pixels
        comp = np.zeros([hist_vec.__len__(), 1])
        alpha = 0.7  # Vad vill jag ha här?
        for ind in pix_vals[0:256]:
            comp[ind] = alpha * cum[ind] + (1 - alpha) * ind
        return hist_vec, cum

    def comp_hist(img, cum):
        alpha = 1
        nbr_rows = img.shape[0]
        nbr_cols = img.shape[1]
        comp_img = np.zeros(img.shape)
        cum = cum * 255  # rescaling from 0-1 to 0-255
        for row in np.arange(0, nbr_rows):
            for col in np.arange(0, nbr_cols):
                intensity = img[row, col]
                comp_img[row, col] = alpha * cum[intensity] + (1 - alpha) * intensity
        return comp_img

    hist_lum, cum_lum = hist_cum_comp(img_lum)
    comp_img = comp_hist(img_lum, cum_lum)
    img_lum[img_lum == 0] = 1
    lum_ratio = comp_img / img_lum
    new_img_col = convert_to_color_img(lum_ratio, img_color)
    imgs = [img_color, new_img_col]
    titles = ['Original', 'Equalized']
    plot_result(imgs, titles)
    return new_img_col, comp_img


def convert_to_color_img(lum_ratio, img_color):
    av_r = img_color[:, :, 0] * lum_ratio
    av_g = img_color[:, :, 1] * lum_ratio
    av_b = img_color[:, :, 2] * lum_ratio
    av_r[av_r > 255] = 255
    av_g[av_g > 255] = 255
    av_b[av_b > 255] = 255
    new_img_col = copy.deepcopy(img_color)
    new_img_col[:, :, 0] = av_r
    new_img_col[:, :, 1] = av_g
    new_img_col[:, :, 2] = av_b
    return new_img_col


def median_filter(img):
    # Selects the median value from each pixels neighborhood
    dims = img.shape.__len__()
    nbr_rows = img.shape[0]
    nbr_cols = img.shape[1]
    median_img = np.zeros(img.shape)
    for row in np.arange(1, nbr_rows-1): # I'll wait with the border
        for col in np.arange(1, nbr_cols-1):
            if dims == 3:
                neighbors = img[row-1:row+2, col-1:col+2, :]
                median_img[row, col, 0] = np.nanmedian(neighbors[0])
                median_img[row, col, 1] = np.nanmedian(neighbors[1])
                median_img[row, col, 2] = np.nanmedian(neighbors[2])
            else:
                neighbors = img[row - 1:row + 2, col - 1:col + 2]
                median_img[row, col] = np.nanmedian(neighbors)

    return median_img


def bilateral_filter(img):

    def filter(img):
        # Define sigmas as the squared versions to not have to square them each time
        sigma_d = 1000  # Size of the considered neighborhood
        sigma_r = 0.05  # "Minimum amplitude of an edge"
        nbr_rows = img.shape[0]
        nbr_cols = img.shape[1]
        img_filt = np.zeros(img.shape)

        nb_size = 3  # Size of the neighborhood to look in each direction of the pixel
        for i in np.arange(nb_size, nbr_rows-nb_size): # Don't care about the border atm
            for j in np.arange(nb_size, nbr_cols-nb_size):
                sum_nom = 0
                sum_den = 0
                for k in np.arange(i-nb_size, i+nb_size+1):
                    for l in np.arange(j-nb_size, j+nb_size+1):
                        exp1 = np.power((i-k), 2) + np.power((j-l), 2)
                        exp2 = np.power((img[i, j] - img[k, l])/256, 2) # Normalized to intensities from 0-1
                        weight = np.exp(-exp1/(2*sigma_d) - exp2/(2*sigma_r))
                        sum_nom = img[k, l]*weight + sum_nom
                        sum_den = weight + sum_den
                img_filt[i, j] = sum_nom / sum_den
        return img_filt

    if img.shape.__len__() == 3:
        new_img = copy.deepcopy(img)
        new_img[:, :, 0] = filter(img[:, :, 0])
        new_img[:, :, 1] = filter(img[:, :, 1])
        new_img[:, :, 2] = filter(img[:, :, 2])
        return new_img
    else:
        return filter(img)


def gaussian_filter(img):

    def filter(img):
        new_img = copy.deepcopy(img)
        gaussian_kernel = 1 / 16 * np.array([1, 4, 6, 4, 1])
        for row in np.arange(0, img.shape[0]):
            new_img[row, :] = np.convolve(img[row, :], gaussian_kernel, mode='same')
        for col in np.arange(0, img.shape[1]):
            new_img[:, col] = np.convolve(new_img[:, col], gaussian_kernel, mode='same')
        return new_img

    if img.shape.__len__() == 3:
        new_img = copy.deepcopy(img)
        new_img[:, :, 0] = filter(img[:, :, 0])
        new_img[:, :, 1] = filter(img[:, :, 1])
        new_img[:, :, 2] = filter(img[:, :, 2])
        return new_img
    else:
        return filter(img)


def main():
# Implement some softening, sharpening, non-linear diffusion (selective sharpening or
# noise removal) filters, such as Gaussian, median, and bilateral (section 3.3.1) as
# discussed in section 3.4.4.

# Take blurry or noisy images (shooting in low light is a good way to get both) and
# try to improve their appearance and legibility.
    img_file = 'original_smaller.jpg'
    img_color = scipy.ndimage.imread(img_file, mode='RGB')
    img_lum = scipy.ndimage.imread(img_file, mode='L')
    imgs = [img_color, img_lum]
    titles = ['original', 'luminance']
    plot_result(imgs, titles)
    img_eq_color, img_eq_lum = hist_eq(img_lum, img_color)

    img_filt = gaussian_filter(img_eq_color)
    img_filt2 = gaussian_filter(img_filt)
    plot_result([img_eq_color, img_filt, img_filt2], ['Equalized', 'Gaussian Filtered', 'Double filtered'])

    img_filt = bilateral_filter(img_eq_color)
    plot_result([img_eq_color, img_filt], ['Equalized', 'Bilateral Filtered'])

    # Final result
 #   plot_result([img_color, img_eq_color, img_filt2], ['Original image', 'Histogram equalized', 'Filtered'])

def main2():

    # Median filtering
    img_filt = median_filter(img_eq_lum)
    images = [img_eq_lum, img_filt]
    titles = ['Eq Luminance', 'Filtered']
    plot_result(images, titles)
    img_new = convert_to_color_img(img_filt/img_eq_lum, img_eq_color)
    images = [img_eq_color, img_new]
    titles = ['Eq Color', 'Filtered']
    plot_result(images, titles)

    img_filt_col = median_filter(img_eq_color)
    images = [img_eq_color, img_filt_col, img_filt_col[:,:,0], img_filt_col[:,:,1], img_filt_col[:,:,2]]
    titles = ['Img_eq_col', 'Filtered color', 'Filtered Red', 'Filtered green', 'Filtered blue']
    plot_result(images, titles)

    test_img = copy.deepcopy(img_eq_color)
    test_img[:,:,0] = img_filt_col[:,:,0]
    test_img[:,:,1] = img_filt_col[:,:,1]
    test_img[:,:,2] = img_filt_col[:,:,2]
    plot_result([img_eq_color, test_img], ['Original', 'Modified'])


if __name__ == '__main__':
    main()
    plt.show(block=True)