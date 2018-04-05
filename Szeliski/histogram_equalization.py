import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt, cm
import copy

def plot_result(imgs, titles):
    nbr_imgs = imgs.__len__()
    print(nbr_imgs)
    fig, axes = plt.subplots(1, nbr_imgs+1, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()

    for ind in np.arange(0, nbr_imgs):
        ax[ind].imshow(imgs[ind])
        ax[ind].set_title(titles[ind])

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()


def hist_cum_comp(img):
    pix_vals = np.arange(0,257,1)
    hist_vec, bins = np.histogram(img, pix_vals)
    nbr_pixels = img.shape[0] * img.shape[1]
    cum = np.zeros([hist_vec.__len__(), 1])
    for ind in pix_vals[0:256]:
        if ind == 0:
            cum[ind] = hist_vec[ind] / nbr_pixels
        else:
            cum[ind] = cum[ind - 1] + hist_vec[ind] / nbr_pixels
    comp = np.zeros([hist_vec.__len__(), 1])
    alpha = 0.7 # Vad vill jag ha här?
    for ind in pix_vals[0:256]:
        comp[ind] = alpha*cum[ind] + (1-alpha)*ind
    return hist_vec, cum


def comp_hist(img, cum):
    alpha = 1
    nbr_rows = img.shape[0]
    nbr_cols = img.shape[1]
    comp_img = np.zeros(img.shape)
    cum = cum*255 # rescaling from 0-1 to 0-255
    for row in np.arange(0, nbr_rows):
        for col in np.arange(0, nbr_cols):
            intensity = img[row, col]
            comp_img[row, col] = alpha*cum[intensity] + (1-alpha)*intensity
    return comp_img


def main():
    # Exercise 3.6 Histogram equalization
    img_file = 'spring.jpg'
    # Convert the color image to luminance (section 3.1.2)
    img_color = scipy.ndimage.imread(img_file, mode='RGB')
    img_lum = scipy.ndimage.imread(img_file, mode='L')
    print(img_color.shape)
    img_r = img_color[:, :, 0]
    img_g = img_color[:, :, 1]
    img_b = img_color[:, :, 2]
    images = [img_color, img_lum, img_r, img_g, img_b]
    titles = ['Original', 'Luminance', 'Red', 'Green', 'Blue']
    plot_result(images, titles)
    # Compute the histogram, the cumulative distribution and the compensation transfer function (sec. 3.1.4)
    hist_lum, cum_lum = hist_cum_comp(img_lum)
    hist_r, cum_r = hist_cum_comp(img_r)
    hist_g, cum_g = hist_cum_comp(img_g)
    hist_b, cum_b = hist_cum_comp(img_b)
    pix_values = np.arange(0,256,1)
    plt.figure()
    plt.scatter(pix_values, hist_lum,c='black')
    plt.scatter(pix_values, hist_r, c='red')
    plt.scatter(pix_values, hist_g, c='green')
    plt.scatter(pix_values, hist_b, c='blue')
    plt.figure()
    plt.scatter(pix_values, cum_lum,c='black')
    plt.scatter(pix_values, cum_r, c='red')
    plt.scatter(pix_values, cum_g, c='green')
    plt.scatter(pix_values, cum_b, c='blue')

    # Comversion from RGB to Luminance is
    # lum = 0.2126*R + 0.7152*G + 0.0722*B


    # (opt) Try to increase the punch in the image by ensuring that a certain fraction of pixels, 5% are mapped to pure b/w
    # (opt) Limit the local gain f'(I) in the transfer function (more details)
    # Compensate the luminance channel through the lookup table and re-generate the color image using color ratios (2.116)
    comp_img = comp_hist(img_lum, cum_lum)
    pix_vals = np.arange(0, 257, 5)
    hist_vec, dummy = np.histogram(comp_img, pix_vals)
    plt.figure()
    plt.scatter(0,0,c='black')
    plt.plot(pix_vals[0:pix_vals.__len__()-1], hist_vec, c='yellow')
    imgs = [img_lum, comp_img]
    titles = ['Luminance', 'Compensated luminance']
    plot_result(imgs, titles)

    # (2.116) After manipulating the luma through histogram equalization you can multiply each color ratio by
    # the ratio of the new to old luma to obtain an adjusted RGB triplet
    RGB = 0.0 + img_r + img_g + img_b
    RGB[RGB==0] = 1
    img_lum[img_lum == 0] = 1
    lum_ratio = comp_img / img_lum
    av_r = img_r * lum_ratio
    av_g = img_g * lum_ratio
    av_b = img_b * lum_ratio
    # lum_ratio = comp_img / img_lum * 255
    # av_r = img_r / RGB * lum_ratio
    # av_g = img_g / RGB * lum_ratio
    # av_b = img_b / RGB * lum_ratio
    av_r[av_r > 255] = 255
    av_g[av_g > 255] = 255
    av_b[av_b > 255] = 255
    new_img_col = copy.deepcopy(img_color)
    new_img_col[:, :, 0] = av_r
    new_img_col[:, :, 1] = av_g
    new_img_col[:, :, 2] = av_b
    imgs = [img_color, new_img_col]
    titles = ['Original', 'Equalized']
    plot_result(imgs, titles)



    # (opt) Color values that are clipped (=saturated) may appear unnatural when remapped. Fix this!


if __name__ == '__main__':
    main()
    plt.show(block=True)