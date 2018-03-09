import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt, cm
plt.clf()
r = np.random.rand(500, 500)
#plt.imshow(r, cmap=None, interpolation='nearest')

file1 = 'mynt.jpg'

img_gray = scipy.ndimage.imread(file1, mode='L')
img_color = scipy.ndimage.imread(file1, mode='RGB')
print(img_color.shape)
#img_color[100:200,100:200,:]=[0,0,255] # Modify some pixels

#plt.imshow(img_color, cmap=cm.gray, interpolation='nearest')
#plt.show(block=True)

def plotColChannels(img) :
    red_image = np.zeros_like(img)
    green_image = np.zeros_like(img)
    blue_image = np.zeros_like(img)
    red_image[:,:,0] = img[:,:,0]
    green_image[:,:,1] = img[:,:,1]
    blue_image[:,:,2] = img[:,:,2]
    fig, (subplot131, subplot132, subplot133) = plt.subplots(ncols=3)
    subplot131.imshow(red_image)
    subplot132.imshow(green_image)
    subplot133.imshow(blue_image)



#plotColChannels(img_color)

# Histograms
#plt.figure(3)
#plt.hist(img_color.flatten())
#plt.figure(4)
#plt.hist(img_gray.flatten())
#plt.figure(5)

for color, channel in zip('rgb', np.rollaxis(img_color, axis=-1)):
    plt.hist(channel.flatten(),color=color,alpha=0.3)

# Segment background from foreground naive

threshold=170 # Kolla i histogrammet eller använd någon metod för att hitta en bra threshold
#plt.figure(6)
#plt.imshow(img_gray > threshold,cmap=cm.gray)


###### Tutorial part 2 #########

import numpy as np
import scipy.ndimage
from matplotlib import pyplot as plt, cm
plt.clf()

file1 = 'mynt.jpg'
img_gray = scipy.ndimage.imread(file1, mode='L')
img_color = scipy.ndimage.imread(file1, mode='RGB')

# Image filtering
bright_square = np.zeros((7,7),dtype=float)
bright_square[2:5,2:5] = 1
#plt.imshow(bright_square, cmap=cm.gray)

def compare2plots(img1, img2):
    fig, (subplot1, subplot2) = plt.subplots(ncols=2)
    subplot1.imshow(img1, cmap=cm.gray)
    subplot2.imshow(img2, cmap=cm.gray)


from scipy.ndimage import convolve
# mean filter
mean_kernel = 1.0/9.0 * np.ones((3,3))
filtered_img = convolve(img_gray, mean_kernel)
#compare2plots(img_gray,filtered_img)


# downsampling
pixelated = img_gray[::10, ::10] # ::10 = picking out every tenth value
#compare2plots(img_gray,pixelated)

# essential filters
# Gaussian filter
smooth_img = scipy.ndimage.filters.gaussian_filter(pixelated, 1)
#compare2plots(pixelated,smooth_img)

# Basic edge filters # Tänk på att i uint8 är -1 = 255!
# edge = np.convolve(img_gray, np.array([1,-1]), mode='valid') # för endim
pixelated = pixelated.astype(float)
pixelated = bright_square
print(pixelated.dtype) # Changed to floats to not get circle effects
vertical_gradient = pixelated[1:,:] - pixelated[:-1,:]
horizontal_gradient = pixelated[:,1:] - pixelated[:, :-1]
#compare2plots(vertical_gradient,horizontal_gradient)

# make an edge filter by convulution kernels
image = pixelated
h_edge_kernel = np.array([[1],[0],[-1]])
v_edge_kernel = np.array([[1, 0, -1]])
print(h_edge_kernel)
vertical_gradient = convolve(image,h_edge_kernel)
horizontal_gradient = convolve(image,v_edge_kernel)
#compare2plots(vertical_gradient,horizontal_gradient)

from skimage import filters
# sobel filter
#compare2plots(image,filters.sobel(image))
# OBS! Good to smooth image before using difference filters!
#compare2plots(smooth_img,filters.sobel(smooth_img))


## Feature detection
# Canny edge detector
#   - Gaussian filter,
#   - sobel filter,
#   - non-maximal suppression, (suppress gradients that are close to stronger gradients)
#   - Hysteresis thresholding (prefer pixels that are connected to edges)

from skimage import data
image = data.camera()
pixelated = image[::10, ::10]
gradient = filters.sobel(pixelated)
#compare2plots(pixelated,gradient)

from skimage import feature
image = data.coins()
edges1 = feature.canny(image,1,1,20) # High and low threshold - islands high, stretching low
edges2 = feature.canny(image,2)
edges3 = feature.canny(image,3)
#compare2plots(image,edges1)
#compare2plots(edges2,edges3)


###### Part 3 of the tutorial ###########
image = data.coins()[0:95,180:370]
plt.figure(1)
plt.imshow(image, cmap=cm.gray,);
edges = feature.canny(image,sigma=3, low_threshold=10, high_threshold=60)
plt.figure(2)
plt.imshow(edges,cmap=cm.gray)
from skimage.transform import hough_circle
hough_radii = np.arange(15,30,2)
hough_response = hough_circle(edges, hough_radii)


## Morphological operations
plt.rcParams['image.cmap'] = 'cubehelix'
plt.rcParams['image.interpolation'] = 'none'

from skimage import morphology
image = np.array([[0,0,0,0,0,0,0],
                  [0,0,0,0,0,0,0],
                  [0,0,1,1,1,0,0],
                  [0,0,1,1,1,0,0],
                  [0,0,1,1,1,0,0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]],dtype=np.uint8)
plt.imshow(image)
sq = morphology.square(width=3)
dia = morphology.diamond(radius=3)
disk = morphology.disk(radius=10)
compare2plots(sq,disk)
compare2plots(image,morphology.erosion(image,sq))
compare2plots(image,morphology.dilation(image,sq))
# opening erosion + dilation
# closing dilation + erosion

from skimage import data,color
hub = color.rgb2gray(data.hubble_deep_field()[350:450, 90:190])
# Plocka bort allt "brus" dvs små saker, men ha kvar den stora
compare2plots(hub,morphology.dilation(morphology.erosion(hub,disk),disk))
compare2plots(hub,morphology.opening(hub,disk))

# Segmentation
# Two different versions contrast-based and boundary-based
# SLIC K-means clustering
from skimage import io, segmentation as seg, color
#url = 'images/spice_1.jpg'
url = 'mynt.jpg'
image = io.imread(url)
labels = seg.slic(image, n_segments=10, compactness=20, sigma = 2, enforce_connectivity=True)
compare2plots(image, labels.astype(float)/labels.max())

from skimage.morphology import watershed
# Watershed är ett annat sätt att segmentera, men vet inte om det tillför så mycket?
# De hävdar också att både SLIC och watershed är för enkla för att använda som final segmentation ouputs.
# Resultaten från dem kallas vanligen superpixels och används sedan för further processing.


## Part 4 of Tutorial
from skimage import io
image = io.imread('chromosomes.jpg')
protein, centromeres, chromosomes = image.transpose((2, 0, 1))
from skimage.filters import threshold_otsu
centromeres_binary = centromeres > threshold_otsu(centromeres)
compare2plots(centromeres, centromeres_binary)

chromosomes_binary = chromosomes > threshold_otsu(chromosomes)
compare2plots(chromosomes, chromosomes_binary)
# Lets try adaptive threshold
from skimage.filters import threshold_adaptive
chromosomes_adapt = threshold_adaptive(chromosomes,block_size=51)
compare2plots(chromosomes, chromosomes_adapt)

img1 = morphology.opening(chromosomes_adapt,morphology.square(4))
img12 = morphology.opening(chromosomes_adapt, morphology.selem.diamond(4))
compare2plots(chromosomes_adapt,img1)
compare2plots(img1,img12)
img2 = morphology.remove_small_objects(img12.astype(bool),300)
compare2plots(img1,img2)

##### Leave last to plot everything
plt.show(block=True)

help(morphology.diamond())