import cv2
import numpy as np
import sys
from matcher_utils import knn_matcher
from image import Image, warp_two_images, read_images, sort_images
from ransac import ransac


def stitch_two_images(img1, img2):
    """Returns stiched image obtained by stiching img1 (on left) and img2 (on right)

    Arguments:
        img1 {Image} -- Image to be stiched on the left
        img2 {[type]} -- Image to be stiched on the right

    Returns:
        Image -- [description]
    """

    # Fetch the matches
    matches = knn_matcher(img1.descriptors, img2.descriptors)

    # Compute the homography using ransac
    h = ransac(matches, img1.keypoints, img2.keypoints,
               sample_points=10, iterations=10000, check=False)

    # Stich two images
    result = Image(warp_two_images(img2.img, img1.img, h))

    return result


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please enter the directory. Correct format --> python stitch.py ../data/")
        sys.exit(0)
    directory = sys.argv[1]
    print(directory)
    images = read_images(directory)
    images = sort_images(images)

    if len(images) == 2:
        stitched = stitch_two_images(images[0], images[1])
        stitched.show()
    elif len(images) == 3:
        stitched = stitch_two_images(images[0], images[1])
        stitched = stitch_two_images(stitched, images[2])
        stitched.show()
