import cv2
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from image_utils import get_sift_features
from matcher_utils import knn_matcher
from ransac import ransac


class Image:
    def __init__(self, img):
        self.img = img
        self.shape = img.shape
        self.keypoints, self.descriptors = get_sift_features(img)

    def show_features(self):
        img = cv2.drawKeypoints(image=self.img, keypoints=self.keypoints,
                                outImage=None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.show()

    def show(self, shape=(15, 15), img_type="bgr"):
        plt.rcParams["figure.figsize"] = shape
        if img_type == "bgr":
            plt.imshow(self.img[:, :, [2, 1, 0]])
        elif img_type.lower() == "gray":
            plt.imshow(self.img, cmap='gray')
        plt.show()


def sort_images(images):
    """Returns a ordered list of Images according to their position in the panorama

    Arguments:
        images {List(Images)} -- List of images which have to be sorted
    """

    if len(images) <= 1:
        return

    num_images = len(images)
    matches_matrix = np.zeros((num_images, num_images))
    max_x_matrix = [[None for _ in range(num_images)]
                    for _ in range(num_images)]

    # Compute number of matches for all possible pairs
    for i in range(len(images)):
        for j in range(i+1, len(images)):
            matches = knn_matcher(images[i].descriptors, images[j].descriptors)
            matches_matrix[(i, j), (j, i)] = len(matches)
            _, img1_x, img2_x = ransac(matches, images[i].keypoints, images[j].keypoints,
                                       sample_points=10, iterations=10000, check=False, return_max_x=True)
            max_x_matrix[i][j] = (img1_x, img2_x)
            max_x_matrix[j][i] = (img2_x, img1_x)

    images_ordered = None

    if len(images) == 3:
        # Image in the middle will be the one with most matches
        middle = np.argmax(matches_matrix.sum(axis=1))

        # Replace None with a tuple for doing numpy operations
        max_x_matrix[middle][middle] = (0, 0)

        # Left image will be the image with highest x coordinate
        left = np.array(max_x_matrix[middle]).argmax(axis=0)[0]

        # Get the right image index
        right = 3 - middle - left

        # Re-order the images
        images_ordered = [images[left], images[middle], images[right]]

    elif len(images) == 2:
        left = 0 if max_x_matrix[1][0][0] > max_x_matrix[1][0][1] else 1
        right = 1 - left
        images_ordered = [images[left], images[right]]
    return images_ordered


def read_images(directory):
    """Returns a List of unique Images from the given directory

    Arguments:
        directory {str} -- Path which has the images

    Returns:
        List(Images) -- List of all Images in the given directory
    """

    images = []
    for file_name in glob(directory + "*"):
        if "panorama" in file_name:
            continue
        read_image = cv2.imread(file_name)
        present_flag = False
        for img in images:
            # Duplicate filter
            if read_image.shape == img.shape and (read_image == img).all():
                present_flag = True
                break
        if not present_flag:
            print("Reading %s" % file_name)
            images.append(read_image)
    
    images = [Image(image) for image in images]
    return images

# Reference : https://stackoverflow.com/questions/13063201/how-to-show-the-whole-image-when-using-opencv-warpperspective


def warp_two_images(img1, img2, h):
    """Warps the given image img1 to img2 using homographt matrix h.

    Arguments:
        img1 {np.ndarray} -- Image to be warped
        img2 {np.ndarray} -- Image to which img1 has to be warped
        h {np.ndarray} -- Homography matrix

    Returns:
        np.ndarray -- Image resulting after warping img1 to img2
    """

    # Find the shape
    y1, x1 = img1.shape[:2]
    y2, x2 = img2.shape[:2]

    # Get the corners of the images
    pts1 = np.array([[0, 0], [0, y1], [x1, y1], [x1, 0]],
                    dtype=np.float32).reshape(-1, 1, 2)
    pts2 = np.array([[0, 0], [0, y2], [x2, y2], [x2, 0]],
                    dtype=np.float32).reshape(-1, 1, 2)

    # Get the points after applying homography
    pts2_ = cv2.perspectiveTransform(pts2, h)
    pts = np.concatenate((pts1, pts2_), axis=0)

    # Round off the points
    [xmin, ymin] = np.array(pts.min(axis=0).flatten() - 0.5, dtype=np.int32)
    [xmax, ymax] = np.array(pts.max(axis=0).flatten() + 0.5, dtype=np.int32)

    # Compute the translation homography matrix
    warped_img = [-xmin, -ymin]
    h_dash = np.array(
        [[1, 0, warped_img[0]], [0, 1, warped_img[1]], [0, 0, 1]])  # translate

    # Use the translation homography matrix to fit bot img1 and img2 on mossiac
    result = cv2.warpPerspective(img2, h_dash.dot(h), (xmax-xmin, ymax-ymin))
    result[warped_img[1]:y1+warped_img[1],
           warped_img[0]:x1+warped_img[0]] = img1

    return result
