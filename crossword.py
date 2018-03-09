import numpy as np
from skimage import io, feature
import scipy.ndimage
import copy
from matplotlib import pyplot as plt, cm
import itertools
import matplotlib.patches as patches

from sklearn import decomposition, neighbors


def compare2plots(img1, img2):
    fig, (subplot1, subplot2) = plt.subplots(ncols=2)
    subplot1.imshow(img1, cmap=cm.gray)
    subplot2.imshow(img2, cmap=cm.gray)


def finding_lines(image, edges):

    from skimage.transform import probabilistic_hough_line
    print(image.shape)
    # Line finding using the Probabilistic Hough Transform

    lines = probabilistic_hough_line(edges, threshold=10, line_length=min(image.shape[0],image.shape[1])*0.2,
                                     line_gap=4)

    # Generating figure
    fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    print("got here")
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
    pairs = itertools.combinations(lines,2)
    for line1, line2 in pairs:
        p1, p2 = line1
        p3, p4 = line2
        i_point = get_intersect(p1, p2, p3, p4)
        if i_point[0] == float('inf'):
            continue
        i_points.append(i_point)
    print('img shape:', image.shape)
    print('corners: ', i_points.__len__())

    i_points_clean = clean_up_points(i_points)
    i_points = np.array(i_points_clean)
    plt.scatter(i_points[:,0],i_points[:,1],s=10)
    ax[3].set_title('Corner points')

    for a in ax:
        a.set_axis_off()
        a.set_adjustable('box-forced')

    plt.tight_layout()
    #plt.show()
    return lines, i_points

def clean_up_points(points):
    nbr_of_points = points.__len__()
    point_pairs = itertools.combinations(points,2)
    to_remove = {} # Creates a dictonary
    threshold = 10
    ok_points = 0
    bad_points = 0
    nbr_of_pairs = 0
    for p1, p2 in point_pairs:
        nbr_of_pairs = nbr_of_pairs +1
        x1, y1 = p1
        x2, y2 = p2
        if p1 in to_remove:
            new_p1 = to_remove.get(p1)
            x1, y1 = new_p1
            #print(new_p1)
        if p2 in to_remove:
            new_p2 = to_remove.get(p2)
            x2, y2 = new_p2
            #print(new_p2)
        # Skapa ett uppslagsverk istället!
            # Kolla om p1 eller p2 finns i uppslagsverket
            # isf ersätt med ny punkt
            # gör kollen om nära varandra
            # isf lägg till i uppslagsverket (key=old_p, value = new_p)
            # när klart, ersätt punkter i listan med de nya från uppslagsverket
        if abs(x1-x2) < threshold:
            if abs(y1-y2) < threshold:
                bad_points = bad_points + 1
                new_point = (int((x1+x2)/2),int((y1+y2)/2))
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
    # print('After removing and adding', new_length)
    if nbr_of_points == new_length:
        return points
    else: clean_up_points(points)
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
    xmin = min(x1,x2)
    xmax = max(x1,x2)
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    if xmin <= x/z <= xmax:
        return (x/z, y/z)
    return (float('inf'),float('inf'))


def old_finding_squares(img):
    [nbr_of_rows, nbr_of_cols] = img.shape
    print(img.shape)
    # Horizontal line
    f1 = img.sum(axis=0) > (nbr_of_cols * 0.3)
    f1 = np.flatnonzero(f1)
    to_remove = np.array([])
    for line_pos in np.nditer(f1[0:-1]):
        ind = np.flatnonzero(f1 == line_pos)
        if abs(line_pos - f1[ind + 1]) < 10:
            to_remove = np.hstack([to_remove, ind])
    f1 = np.delete(f1, to_remove)  # Removing double-lines
    print('Shape f1: ', f1.shape)

    # Vertical line
    f2 = img.sum(axis=1) > (nbr_of_rows * 0.3)
    f2 = np.flatnonzero(f2)
    to_remove = np.array([])
    for line_pos in np.nditer(f2[0:-1]):
        ind = np.flatnonzero(f2 == line_pos)
        if abs(line_pos - f2[ind + 1]) < 10:
            to_remove = np.hstack([to_remove, ind])
    f2 = np.delete(f2, to_remove)  # Removing double-lines
    print('Shape f2: ', f2.shape)

    # Corners
    corners = np.zeros(img.shape)
    corners[np.ix_(f2, f1)] = 1
    # compare2plots(img,corners)

    square_points = np.array([0, 0, 0, 0],dtype=int)
    for corner_row in np.nditer(f1[0:-1]):
        ind_row = np.flatnonzero(f1 == corner_row)
        for corner_col in np.nditer(f2[0:-1]):
            ind_col = np.flatnonzero(f2 == corner_col)
            new_square = np.array([corner_row, f1[ind_row+1], corner_col, f2[ind_col+1]],dtype=int)
            square_points = np.vstack([square_points, new_square])
    print(square_points.shape)
    return square_points

def finding_squares(img, img_col):
    lines, corner_points = finding_lines(img_col, img)
    squares=np.array([0, 0, 0, 0],dtype=int)
    for corner in corner_points:
        upp_right = closest_right(corner, corner_points)
        if upp_right.any() == None:
           # print('no point to the right of', corner)
            continue
        down_left = closest_down(corner, corner_points)
        if down_left.any() == None:
           # print('no point below', corner)
            continue
        square = np.array([corner[0],upp_right[0],corner[1],down_left[1]],dtype=int)
        squares = np.vstack([squares, square])
    print('nbr of squares', squares.__len__())
    return squares[1:,:]

def closest_down(corner, corner_points):
    closest_dist = 100000
    alignment_threshold = 20
    closest_down = np.array([None, None])
    for c in corner_points:
        if c[1] > corner[1]:
            if abs(c[0]-corner[0]) < alignment_threshold:
                if c[1] - corner[1] < closest_dist:
                    closest_dist = c[1]-corner[1]
                    closest_down = c
    return closest_down

def closest_right(corner, corner_points):
    closest_dist = 100000
    alignment_threshold = 20
    closest_right = np.array([None, None])
    for c in corner_points:
        if c[0] > corner[0]:
            if abs(c[1]-corner[1]) < alignment_threshold:
                if c[0] - corner[0] < closest_dist:
                    closest_dist = c[0]-corner[0]
                    closest_right = c
    return closest_right





def color_square(nbr, col_code, img, square_points):
    [row1, row2, col1, col2] = square_points[nbr, :]
    # print(row1,row2,col1,col2)
    img[col1:col2, row1:row2] = col_code  # Array indexing x1:x2,y1:y2


def sq_points_2_X(sq, img):
    # Gets a matrix with the corner points of all squares in the image
    # and the original grayscale image

    f_sq = sq[100, :]
    print(f_sq)
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
    return [X[1:,:], STANDARD_SIZE]


def classify_squares_features(X, STANDARD_SIZE):
    nbr_of_features = 7
    features = np.zeros([X.shape[0], nbr_of_features])
    total_nbr_pixels = STANDARD_SIZE[0]*STANDARD_SIZE[1]
    print('Pixels: ', total_nbr_pixels)
    print('Image size', STANDARD_SIZE)

    for ind in np.arange(0, X.shape[0]):
        img_el = np.reshape(X[ind, :], STANDARD_SIZE)
        # plt.figure()
        # plt.imshow(img_el)
        # print(img_el)

    # feature 1 - average pixel intensity
        f1 = img_el.sum() / total_nbr_pixels #/ 255 #normalized to a number from 0-1
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
        features[ind,:] = [f1, f2, f3, f4, f5, f6, f7]
    return features


def classify_squares_train(X, y):

    pca = decomposition.PCA(n_components=5, svd_solver='randomized')
    # Vet inte skillnaden på randomized och inte, borde kolla upp
    X = pca.fit_transform(X)
    print(X.shape)
    print('yshape', y.shape)

    plotting = False
    if plotting:
        plt.figure(4)
        plt.scatter(X[:, 0], X[:, 1], c=y)
        plt.colorbar()
        plt.figure(5)
        plt.scatter(X[:, 2], X[:, 3], c=y)
        plt.colorbar()
        plt.figure(6)
        plt.scatter(X[:, 0], X[:, 4], c=y)
        plt.colorbar()

    knn = neighbors.KNeighborsClassifier()
    knn.fit(X, y)
    return [pca, knn]


def classify_squares_predict(X, knn):
    preds = knn.predict(X)
    return preds
    # Want to classify the squares into one of the following
    # 0 = Empty square
    # 1 = Lead square
    # 2 = Filled square
    # 3 = Image/Other/unknown


def set_targets(img_color, squares):

    class LineBuilder:
        def __init__(self, line):
            self.line = line
            self.xs = list(line.get_xdata())
            self.ys = list(line.get_ydata())
            self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
            self.target_vec = np.zeros(squares.shape[0],dtype=int)

        def __call__(self, event):
            print('click', event)
            xcord = event.xdata
            ycord = event.ydata
            ind, sq = find_square(xcord, ycord)
            target_vec = toggle_square(ind, sq, self.target_vec)
            #target = input('Target this square 0-3: ')
            self.line.set_data(self.xs,self.ys)
            self.line.figure.canvas.draw()

        def get_target_vec(self):
            return self.target_vec

    def toggle_square(ind, sq, targets):
        if targets[ind] == 3:
            targets[ind] = 0
        else:
            targets[ind] = int(targets[ind]+1)
        print(int(targets[ind]))
        colors = ["white","green","red","blue"]
        ax.add_patch(patches.Rectangle((sq[0], sq[2]), sq[1] - sq[0], sq[3] - sq[2], alpha=0.5, fc=colors[targets[ind]]))
        return targets


    def find_square(x,y):
        for sq_ind in np.arange(0,squares.shape[0]):
            sq = squares[sq_ind,:]
            if sq[0]< x < sq[1] and sq[2] < y < sq[3]:
                return sq_ind, sq
        print('No square found!')


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img_color)
    ax.set_title('Classify the squares by clicking on them; white=empty, green=lead, red=letter, blue=other')
    line, = ax.plot([0], [0])  # empty line
    linebuilder = LineBuilder(line)

    plt.show()
    return LineBuilder.get_target_vec(linebuilder)

def get_targets(img_color,squares): # Just a dummy right now, don't know the order of the squares...
    targets = set_targets(img_color,squares)
    print(targets)
    #targets = np.random.randint(0,4,squares.shape[0])
    return targets


def read_process_image(img_file):
    image = io.imread(img_file, as_grey=True)
    img_gray = scipy.ndimage.imread(img_file, mode='L')
    img_color = scipy.ndimage.imread(img_file, mode='RGB')

    # plt.hist(image)
    #edges = feature.canny(image, sigma=2, low_threshold=0, high_threshold=0.1)
    edges = feature.canny(image, sigma=0.5, low_threshold=0, high_threshold=0)

    return [img_gray, img_color, edges]


def plot_result(img, preds, squares, title):
    img_prediction = copy.deepcopy(img)
    for sq_nbr in np.flatnonzero(preds == 3):
        color_square(sq_nbr, [0,0,250], img_prediction, squares)

    for sq_nbr in np.flatnonzero(preds == 1):
        color_square(sq_nbr, [0,250,0], img_prediction, squares)
    compare2plots(img, img_prediction)
    plt.title(title)


def main():
    img_file = 'DN.Korsord.jpg'
    [img_gray, img_color, edges] = read_process_image(img_file)
    squares = finding_squares(edges, img_color)
    print('nbr of squares returned', squares.shape)
    targets = get_targets(img_color, squares)

    [X, STANDARD_SIZE] = sq_points_2_X(squares, img_gray)

    feature_vec = classify_squares_features(X, STANDARD_SIZE)
    knn = neighbors.KNeighborsClassifier(n_neighbors=10)
    knn.fit(feature_vec, targets)
    preds = classify_squares_predict(feature_vec, knn)

    # Prediction results
    plot_result(img_color, preds, squares, 'Training data features')
    plot_squares(img_color, squares)


    ############ PCA ##############
    # PCA of feature-vector
    knn_pca = neighbors.KNeighborsClassifier(n_neighbors=10)
    pca = decomposition.PCA(n_components=1, svd_solver='randomized')
    pcaX = pca.fit_transform(feature_vec)
    knn_pca.fit(pcaX, targets)
    preds_pca = classify_squares_predict(pcaX, knn_pca)
    # Prediction results
    plot_result(img_color, preds_pca, squares, 'PCA on feature vector')
    ###############################



    # Validation data
    img_file = 'DN.Korsord2.jpg'
    [img_gray2, img_color2, edges2] = read_process_image(img_file)
    squares2 = finding_squares(edges2, img_color2)
    [X, STANDARD_SIZE] = sq_points_2_X(squares2, img_gray2)
    feature_vec = classify_squares_features(X, STANDARD_SIZE)
    preds2 = classify_squares_predict(feature_vec, knn)
    plot_result(img_color2, preds2, squares2, 'Validation')

    pcaX = pca.fit_transform(feature_vec)
    preds_pca = classify_squares_predict(pcaX, knn_pca)
    plot_result(img_color2, preds_pca, squares2, 'Validation PCA features')


def main_pca():
    img_file = 'DN.Korsord.jpg'
    [img_gray, img_color, edges] = read_process_image(img_file)
    squares = finding_squares(edges)
    targets = get_targets(img_color, squares)
    [X, STANDARD_SIZE] = sq_points_2_X(squares, img_gray)
    print('X1', X.shape)
    [pca, knn] = classify_squares_train(X, targets)
    pcaX = pca.fit_transform(X)
    preds = classify_squares_predict(pcaX, knn)
    plot_result(img_color, preds, squares, 'Training data PCA')

    # Validation
    img_file = 'DN.Korsord2.jpg'
    [img_gray2, img_color2, edges2] = read_process_image(img_file)
    squares2 = finding_squares(edges2)
    [X, STANDARD_SIZE] = sq_points_2_X(squares2, img_gray2)
    print('X2', X.shape)
    pcaX = pca.fit_transform(X)
    preds2 = classify_squares_predict(pcaX, knn)
    plot_result(img_color2, preds2, squares2, 'Validation data PCA')

    ####### Plotting results ##########
#    plotting_squares = False
#    if plotting_squares:
#        plot_squares(img_color, squares)
#        plot_squares(img_color2, squares2)


def plot_squares(img, squares):
    img_squares = copy.deepcopy(img)  # To not only make a new reference binding to the same object
    for sq_nbr in np.arange(0, squares.shape[0] - 3, 4):
        color_square(sq_nbr, [1, 100, 1], img_squares, squares)
        color_square(sq_nbr + 1, [100, 1, 100], img_squares, squares)
        color_square(sq_nbr + 2, [1, 1, 100], img_squares, squares)
        color_square(sq_nbr + 3, [100, 1, 1], img_squares, squares)
    compare2plots(img, img_squares)


if __name__ == '__main__':
    #get_intersect_obs((1,1),(1.2,1.2),(1,2),(2,1))
    main()
    # main_pca()
    plt.show(block=True)