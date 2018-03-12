import numpy as np
from skimage import io, feature
import scipy.ndimage
import copy
from matplotlib import pyplot as plt, cm
import itertools
import matplotlib.patches as patches
import pickle
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
    pairs = itertools.combinations(lines, 2)
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
    plt.scatter(i_points[:,0], i_points[:, 1], s=10)
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
    squares = squares[1:,:]
    print('nbr of squares', squares.__len__())
    return squares


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
            self.target_vec = toggle_square(ind, sq, self.target_vec)
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


def get_targets(img_color,squares):
    targets = set_targets(img_color,squares)
    print(targets)
    return targets


def read_process_image(img_file):
    image = io.imread(img_file, as_grey=True)
    img_gray = scipy.ndimage.imread(img_file, mode='L')
    img_color = scipy.ndimage.imread(img_file, mode='RGB')
    edges = feature.canny(image, sigma=0.5, low_threshold=0, high_threshold=0)
    return [img_gray, img_color, edges]


def main():
    img_file = 'DN.Korsord.jpg'
    [img_gray, img_color, edges] = read_process_image(img_file)
    squares = finding_squares(edges, img_color)
    print('nbr of squares returned', squares.shape)
    targets = get_targets(img_color, squares)

    with open('training_data.pkl', 'wb') as f:
        pickle.dump([squares, targets, img_gray], f)
    with open('training_data.pkl', 'rb') as f:
        obj0, obj1, obj2 = pickle.load(f)
    print(obj0.shape, obj1.shape)

    [X, STANDARD_SIZE] = sq_points_2_X(squares, img_gray)



if __name__ == '__main__':
    main()
    plt.show(block=True)