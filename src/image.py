import cv2
import numpy as np
import matplotlib.pyplot as plt

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def print_matches(matches):
    try:
        if type(matches[0]) not in (np.ndarray, list):
            for match in matches:
                print('Train=%4d Query=%4d Distance=%4.2f' %
                      (match.trainIdx, match.queryIdx, match.distance))
        else:
            for match_ in matches:
                for match in match_:
                    print('Train=%4d Query=%4d Distance=%4.2f' %
                          (match.trainIdx, match.queryIdx, match.distance))
    except:
        pass


def draw_matches(img1, img2, kp1, kp2, matches):
    img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                           matches, outImg=np.array([]))  # , matchColor=(0, 255, 0))
    show_image(img3)

# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html


def get_sift_features(img):
    if len(img.shape) == 3:
        img = to_grayscale(img)
    sift_obj = cv2.xfeatures2d.SIFT_create()
    return sift_obj.detectAndCompute(img, None)


'''
    Gets matches for arr2 in arr1
'''
def knn_matcher(arr2, arr1, neighbours=2, img_id=0, ratio_threshold = 0.75):
    assert neighbours == 2

    all_distances = np.sqrt(np.square(arr2).sum(
        axis=1)[:, np.newaxis] + np.square(arr1).sum(axis=1) - 2 * arr2.dot(arr1.T))

    closest_indices = np.argsort(all_distances, axis=1)[:, :neighbours]

    matches = []
    for i in range(closest_indices.shape[0]):
        match_list = [cv2.DMatch(
            _trainIdx=n, _queryIdx=i, _distance=all_distances[i, n], _imgIdx=img_id) for n in closest_indices[i]]
        matches.append(match_list)
    if ratio_threshold:
        matches = filter_matches(matches, ratio_threshold)
    
    return matches


def filter_matches(matches, threshold = 0.75):
    filtered = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            filtered.append(m)
    return filtered


'''
    Computes the homography matrix for the given points using 
    the Direct Linear transform
'''
def get_homography_matrix(pts1, pts2):
    eqn_list = []
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        row1 = [x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2]
        row2 = [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2]
        eqn_list.append(row1)
        eqn_list.append(row2)

    # Solve for A.h = 0
    A = np.array(eqn_list)
    eig_values, eig_vectors = np.linalg.eig(np.dot(A.T, A))
    h_index = eig_values.argmin()
    h = eig_vectors[:, h_index]

    # Convert it to a unit vector
    h = h / np.linalg.norm(h)    
    h /= h[-1]
    h = h.reshape(3,3)
    return h


def show_image(img, shape=(15, 15), img_type="bgr"):
    plt.rcParams["figure.figsize"] = shape
    if img_type == "bgr":
        plt.imshow(img[:, :, [2, 1, 0]])
    elif img_type.lower() == "gray":
        plt.imshow(img, cmap='gray')
    plt.show()
    
#https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective
def warpTwoImages(img1, img2, h):
    '''warp img2 to img1 with homograph h'''
    y1,x1 = img1.shape[:2]
    y2,x2 = img2.shape[:2]
    pts1 = np.array([[0,0],[0,y1],[x1,y1],[x1,0]], dtype = np.float32).reshape(-1,1,2)
    pts2 = np.array([[0,0],[0,y2],[x2,y2],[x2,0]], dtype=np.float32).reshape(-1,1,2)
    pts2_ = cv2.perspectiveTransform(pts2, h)
    pts = np.concatenate((pts1, pts2_), axis=0)
    [xmin, ymin] = np.array(pts.min(axis=0).flatten() - 0.5, dtype=np.int32)
    [xmax, ymax] = np.array(pts.max(axis=0).flatten() + 0.5, dtype=np.int32)
    warped_img = [-xmin,-ymin]
    h_dash = np.array([[1,0,warped_img[0]],[0,1,warped_img[1]],[0,0,1]]) # translate

    result = cv2.warpPerspective(img2, h_dash.dot(h), (xmax-xmin, ymax-ymin))
    result[warped_img[1]:y1+warped_img[1],warped_img[0]:x1+warped_img[0]] = img1
    return result
