import numpy as np
from skimage import io, feature
import scipy.ndimage
import copy
from matplotlib import pyplot as plt, cm
import itertools
import pickle
from sklearn import decomposition, neighbors


def finding_lines(image, edges):
    from skimage.transform import probabilistic_hough_line

    # Line finding using the Probabilistic Hough Transform
    lines = probabilistic_hough_line(edges, threshold=10,
                                     line_length=min(image.shape[0], image.shape[1])*0.2,
                                     line_gap=4)
    # Generating figure
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')

    ax[1].imshow(edges, cmap=cm.gray)
    ax[1].set_title('Canny edges')

    ax[2].imshow(edges * 0)
    for line in lines:
        p0, p1 = line
        ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
    ax[2].set_xlim((0, image.shape[1]))
    ax[2].set_ylim((image.shape[0], 0))
    ax[2].set_title('Probabilistic Hough')

    ax[3].imshow(image)
    i_points = []
    pairs = itertools.combinations(lines, 2)
    for line1, line2 in pairs:
        p1, p2 = line1
        p3, p4 = line2
        i_point = get_intersect(p1, p2, p3, p4)
        if i_point[0] == float('inf'):
            continue
        i_points.append(i_point)

    i_points_clean = clean_up_points(i_points)
    i_points = np.array(i_points_clean)
    plt.scatter(i_points[:, 0], i_points[:, 1], s=10)
    ax[3].set_title('Corner points')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()
    plt.show()
    return lines, i_points

def plot_result(img, predictions, squares, titles):
    nbr_preds = predictions.shape[0]
    fig, axes = plt.subplots(1, nbr_preds+1, figsize=(15,5), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(img)
    ax[0].set_title('Input image')

    for ind in np.arange(1, nbr_preds+1):
        img_prediction = copy.deepcopy(img)
        preds = predictions[ind-1, :]
        title = titles[ind-1, :]
        for sq_nbr in np.flatnonzero(preds == 3):
            color_square(sq_nbr, [0, 0, 250], img_prediction, squares)

        for sq_nbr in np.flatnonzero(preds == 1):
            color_square(sq_nbr, [0, 250, 0], img_prediction, squares)

        ax[ind].imshow(img_prediction)
        ax[ind].set_title(title)

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()


def clean_up_points(points):
    nbr_of_points = points.__len__()
    point_pairs = itertools.combinations(points, 2)
    to_remove = {}  # Creates a dictonary
    threshold = 10
    ok_points = 0
    bad_points = 0
    nbr_of_pairs = 0
    for p1, p2 in point_pairs:
        nbr_of_pairs = nbr_of_pairs + 1
        x1, y1 = p1
        x2, y2 = p2
        if p1 in to_remove:
            new_p1 = to_remove.get(p1)
            x1, y1 = new_p1
        if p2 in to_remove:
            new_p2 = to_remove.get(p2)
            x2, y2 = new_p2
        if abs(x1-x2) < threshold:
            if abs(y1-y2) < threshold:
                bad_points = bad_points + 1
                new_point = (int((x1+x2)/2), int((y1+y2)/2))
                to_remove[p1] = new_point
                to_remove[p2] = new_point
        ok_points = ok_points + 1
    keys = to_remove.keys()
    for key in keys:
        points.remove(key)
        value = to_remove[key]
        if value in points:
            continue
        points.append(value)
    new_length = points.__len__()
    if nbr_of_points == new_length:
        return points
    else:
        clean_up_points(points)
    return points


def get_intersect(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    x1=a1[0]; x2=a2[0];
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return float('inf'), float('inf')
    if xmin <= x/z <= xmax:
        return x/z, y/z
    return float('inf'), float('inf')


def finding_squares(img, img_col):
    lines, corner_points = finding_lines(img_col, img)
    print('lines', lines.shape)
    squares=np.array([0, 0, 0, 0], dtype=int)
    for corner in corner_points:
        upp_right = closest_right(corner, corner_points)
        if upp_right.any() is None:
            # print('no point to the right of', corner)
            continue
        down_left = closest_down(corner, corner_points)
        if down_left.any() is None:
            # print('no point below', corner)
            continue
        square = np.array([corner[0], upp_right[0], corner[1], down_left[1]], dtype = int)
        squares = np.vstack([squares, square])
    squares = squares[1:, :]
    print('nbr of squares', squares.__len__())
    return squares


def closest_down(corner, corner_points):
    closest_dist = 100000
    alignment_threshold = 20
    cl_down = np.array([None, None])
    for c in corner_points:
        if c[1] > corner[1]:
            if abs(c[0]-corner[0]) < alignment_threshold:
                if c[1] - corner[1] < closest_dist:
                    closest_dist = c[1]-corner[1]
                    cl_down = c
    return cl_down


def closest_right(corner, corner_points):
    closest_dist = 100000
    alignment_threshold = 20
    cl_right = np.array([None, None])
    for c in corner_points:
        if c[0] > corner[0]:
            if abs(c[1]-corner[1]) < alignment_threshold:
                if c[0] - corner[0] < closest_dist:
                    closest_dist = c[0]-corner[0]
                    cl_right = c
    return cl_right


def color_square(nbr, col_code, img, square_points):
    [row1, row2, col1, col2] = square_points[nbr, :]
    img[col1:col2, row1:row2] = col_code  # Array indexing x1:x2,y1:y2


def sq_points_2_X(sq, img):
    # Gets a matrix with the corner points of all squares in the image
    # and the original grayscale image

    f_sq = sq[100, :]
    first_el = img[f_sq[0]:f_sq[1], f_sq[2]:f_sq[3]]
    STANDARD_SIZE = first_el.shape
    print('STANDARD_SIZE', STANDARD_SIZE)
    X = first_el.flatten()
    for sq_ind in np.arange(0, sq.shape[0]):
        [row1, row2, col1, col2] = sq[sq_ind]
        dummy = img[col1:col2, row1:row2]
        img_el = copy.deepcopy(dummy)
        img_el.resize(STANDARD_SIZE)
        X = np.vstack([X, img_el.flatten()])

    print('X is: ', X.shape)
    return [X[1:, :], STANDARD_SIZE]


def classify_squares_features(X, STANDARD_SIZE):
    nbr_of_features = 7
    features = np.zeros([X.shape[0], nbr_of_features])
    total_nbr_pixels = STANDARD_SIZE[0]*STANDARD_SIZE[1]

    for ind in np.arange(0, X.shape[0]):
        img_el = np.reshape(X[ind, :], STANDARD_SIZE)

    # feature 1 - average pixel intensity
        f1 = img_el.sum() / total_nbr_pixels  # / 255 #normalized to a number from 0-1
    # feature 2 - total number of "white" pixels div by total number of pixels
        whites = img_el > 200
        f2 = np.sum(whites) / total_nbr_pixels
    # feature 3 - average of "black" pixels
        blacks = img_el < 50
        f3 = np.sum(blacks) / total_nbr_pixels
    # feature 4 - average of white rows
        f4 = np.sum(whites.sum(axis=1) == 0) / STANDARD_SIZE[0]
    # feature 5 - average of white cols
        f5 = np.sum(whites.sum(axis=0) == 0) / STANDARD_SIZE[1]
    # feature 6 - average of rows with (some) black pixels
        f6 = np.sum(blacks.sum(axis=1) > 0) / STANDARD_SIZE[0]
    # feature 7 - average of cols with (some) black pixels
        f7 = np.sum(blacks.sum(axis=0) > 0) / STANDARD_SIZE[1]
        features[ind, :] = [f1, f2, f3, f4, f5, f6, f7]
    return features


def read_process_image(img_file):
    image = io.imread(img_file, as_grey=True)
    img_gray = scipy.ndimage.imread(img_file, mode='L')
    img_color = scipy.ndimage.imread(img_file, mode='RGB')

    # scale_factor = 1.2
    img_size = (int(image.shape[0]/scale_factor), int(image.shape[1]/scale_factor))
    # image = scipy.ndimage.zoom(image, 0.1)
    #image.zoom(0.5)  #TODO: zoom seems to work much better but this code still doesn't work.
    # #TODO: Looks like resize is doing a terrible job, check what it looks like for the elements before
    # img_gray.resize(img_size)
    # img_color.resize(img_size)
    # print(image.shape)
    # plt.imshow(image)
    # plt.show()
    edges = feature.canny(image, sigma=0.5, low_threshold=0, high_threshold=0)
    print(edges.shape)
    return [img_gray, img_color, edges]


def get_predictor(filename):
    with open(filename, 'rb') as f:
        squares, targets, training_img = pickle.load(f)
    [X, STANDARD_SIZE] = sq_points_2_X(squares, training_img)
    feature_vec = classify_squares_features(X, STANDARD_SIZE)

    # Feature vector
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(feature_vec, targets)

    # First component PCA on feature vector
    knn_pca = neighbors.KNeighborsClassifier(n_neighbors=10)
    pca = decomposition.PCA(n_components=1, svd_solver='randomized')
    pcaX = pca.fit_transform(feature_vec)
    knn_pca.fit(pcaX, targets)

    # PCA only
    pca_only = decomposition.PCA(n_components=5, svd_solver='randomized')
    X = pca_only.fit_transform(X)
    knn_pca_only = neighbors.KNeighborsClassifier()
    knn_pca_only.fit(X, targets)

    return knn, knn_pca, pca, pca_only, knn_pca_only


def classify_crossword(img_file, titleline):

    [img_gray, img_color, edges] = read_process_image(img_file)
    squares = finding_squares(edges, img_color)
    knn, knn_pca, pca, pca_only, knn_pca_only = get_predictor('training_data.pkl')
    [X, STANDARD_SIZE] = sq_points_2_X(squares, img_gray)
    feature_vec = classify_squares_features(X, STANDARD_SIZE)

    ######## Feature vector #########
    preds = knn.predict(feature_vec)
    title = titleline + ' features'
    # plot_result(img_color, preds, squares, titleline + ' features')

    ########## PCA on feature vec ############
    pcaX = pca.fit_transform(feature_vec)
    preds = np.vstack([preds, knn_pca.predict(pcaX)])
    title = np.vstack([title, titleline + ' PCA features'])
    # plot_result(img_color, preds_pca, squares, titleline + ' PCA features')

    ######### Only PCA ##########
    pcaX_only = pca_only.fit_transform(X)
    preds = np.vstack([preds, knn_pca_only.predict(pcaX_only)])
    title = np.vstack([title, titleline + ' PCA only'])

    plot_result(img_color, preds, squares, title)


def main():  # TODO: Make it possible to run file with image file as argument

 #   classify_crossword('DN.Korsord.jpg', 'Training data')
 #   classify_crossword('DN.Korsord2.jpg', 'Validation')
    classify_crossword('bilder/20180313_143109.jpg', 'Half solved')

if __name__ == '__main__':
    main()
    plt.show(block=True)