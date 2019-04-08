import numpy as np
from math import ceil

from matcher_utils import knn_matcher
from homography import get_homography_matrix


def ransac(matches, kp1, kp2, sample_points=4, iterations=5, inlier_tolerance=3, inlier_ratio=0.45, check=True, return_max_x=False):
    """Computes the homography matrix for the given matches using RAndom SAmple Consensus

    Arguments:
        matches {cv.DMatch} -- Matches for which homography matrix has to be computed
        kp1 {list(keypoints)} -- List of keypoints from image 1
        kp2 {[type]} -- List of keypoints from umage 2

    Keyword Arguments:
        sample_points {int} -- Number of point samples to be used as random samples for computing homography (default: {4})
        iterations {int} -- Number of iterations to run (default: {5})
        inlier_tolerance {int} -- acceptable offset for match to be considered inlier (default: {3})
        inlier_ratio {float} -- [description] (default: {0.45})
        check {bool} -- If True then prints (default: {True})
        return_max_x {bool} -- If true, returns the maximum x coordinates of inliers for both image (default: {False})

    Returns:
        np.ndarray -- Homography matrix
        [int] -- maximum x coordinate value for inlier match from kp1 (If return_max_x is True)
        [int] -- maximum x coordinate value for inlier match from kp2 (If return_max_x is True)
    """

    best_inlier_count = 0
    best_h = None
    best_inlier_indices = None

    # Get all the corresponing matching pairs for both the images
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])

    # Re-usable variables for all iterations
    homogeneous_pts1 = np.hstack((pts1, np.ones((pts1.shape[0], 1)))).T
    indices = np.arange(len(pts1))
    num_pts = pts1.shape[0]
    required_inliers = inlier_ratio * num_pts

    # For number of iterations
    for _ in range(iterations):

        # Sample a small set of points from the point match pairs
        indices_to_sample = np.random.choice(indices, sample_points)
        pts1_sample = pts1[indices_to_sample]
        pts2_sample = pts2[indices_to_sample]

        # Get the homography matrix
        h = get_homography_matrix(pts1_sample, pts2_sample)

        # Find the new points using the homography matrix
        transformed_points = np.dot(h, homogeneous_pts1).T

        # Convert it to world coordinates
        last_col = np.copy(transformed_points[:, -1])
        last_col = last_col[:, np.newaxis]
        transformed_points /= last_col
        transformed_points = transformed_points[:, :-1]

        # Find the distance between the actual and the mapped points
        distance = np.linalg.norm(pts2 - transformed_points, axis=1)
        inlier_indices = distance < inlier_tolerance
        inlier_count = inlier_indices.sum()

        # Update the best_h if the current h has more inliers
        if inlier_count > best_inlier_count:
            best_h = h
            best_inlier_indices = inlier_indices
            best_inlier_count = inlier_count

        # If required inliers is reached break
        if inlier_count > required_inliers:
            break

    # Verbose mode - Print the number of inliers
    if check:
        transformed_points = np.dot(best_h, homogeneous_pts1).T
        # Convert it to world coordinates
        last_col = np.copy(transformed_points[:, -1])
        last_col = last_col[:, np.newaxis]
        transformed_points /= last_col
        transformed_points = transformed_points[:, :-1]
        distance = np.linalg.norm(pts2 - transformed_points, axis=1)
        inlier_count = len(distance[distance < inlier_tolerance])
        print('%2.2f of the points are inliers' %
              (inlier_count / num_pts * 100))

    # If x coordinates are needed
    if return_max_x:
        max_x_inlier_1 = ceil(pts1[best_inlier_indices].max(axis=0)[0])
        max_x_inlier_2 = ceil(pts2[best_inlier_indices].max(axis=0)[0])
        return best_h, max_x_inlier_1, max_x_inlier_2
    return best_h
