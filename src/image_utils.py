import cv2
import numpy as np
import matplotlib.pyplot as plt


def to_grayscale(img):
    """Converts a given image to grayscale

    Arguments:
        img {np.ndarray} -- Image which has to be converted to grayscale

    Returns:
        [np.ndarray] -- Grayscale  copy of the given image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def show_image(img, shape=(15, 15), img_type="bgr"):
    """Shows the given image with given shape

    Arguments:
        img {np.ndarray} -- Image to be displayed

    Keyword Arguments:
        shape {tuple} -- Shape of the image to be displayed (default: {(15, 15)})
        img_type {str} -- Type of given image. Can be "bgr" or "gray" (default: {"bgr"})
    """

    plt.rcParams["figure.figsize"] = shape
    if img_type == "bgr":
        plt.imshow(img[:, :, [2, 1, 0]])
    elif img_type.lower() == "gray":
        plt.imshow(img, cmap='gray')
    plt.show()


# https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
def get_sift_features(img):
    """Returns the keypoints and descriptors for a given image

    Arguments:
        img {np.ndarray} -- Image for wich keypoints and descriptors are needed

    Returns:
        tuple(keypoints, descriptors) -- A tuple of keypoints which is a list of keypoints
        and descriptors which are a numpy array
    """

    # Convert image to grayscale
    if len(img.shape) == 3:
        img = to_grayscale(img)
    sift_obj = cv2.xfeatures2d.SIFT_create()
    k, d = sift_obj.detectAndCompute(img, None)
    return k, d



