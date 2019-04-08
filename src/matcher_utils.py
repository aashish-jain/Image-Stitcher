import cv2
import numpy as np
import matplotlib.pyplot as plt


def filter_matches(matches, threshold=0.75):
    """Returns filterd copy of matches grater than given threshold

    Arguments:
        matches {list(tuple(cv2.DMatch))} -- List of tupe of cv2.DMatch objects

    Keyword Arguments:
        threshold {float} -- Filter Threshold (default: {0.75})

    Returns:
        list(cv2.DMatch) -- List of cv2.DMatch objects that satisfy ratio test
    """

    filtered = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            filtered.append(m)
    return filtered


def knn_matcher(arr2, arr1, neighbours=2, img_id=0, ratio_threshold=0.75):
    """Computes the inlier matches for given descriptor ararys arr1 and arr2

    Arguments:
        arr2 {np.ndarray} -- Image used for finding the matches (train image)
        arr1 {[type]} -- Image in which matches are found (test image)

    Keyword Arguments:
        neighbours {int} -- Number of neighbours to consider while matching. 
        Should be 2 (default: {2})
        img_id {int} -- Id of the train image (default: {0})
        ratio_threshold {float} -- Ratio threshold for the ratio test 
        (default: {0.75}). If 0 or None, the mathes are not filtered.

    Returns:
        list(matches) -- List of cv2.DMatch objects
    """

    assert neighbours == 2

    # Compute L2 distance for all the descriptors arr1 and arr2
    all_distances = np.sqrt(np.square(arr2).sum(
        axis=1)[:, np.newaxis] + np.square(arr1).sum(axis=1) - 2 * arr2.dot(arr1.T))

    # Take top K closest neighbours for each descriptor
    closest_indices = np.argsort(all_distances, axis=1)[:, :neighbours]

    # Create a list of "K" match pairs
    matches = []
    for i in range(closest_indices.shape[0]):
        match_list = [cv2.DMatch(
            _trainIdx=n, _queryIdx=i, _distance=all_distances[i, n], _imgIdx=img_id) for n in closest_indices[i]]
        matches.append(match_list)

    # Perform ratio test to get inliers
    if ratio_threshold:
        matches = filter_matches(matches, ratio_threshold)

    return matches


def print_matches(matches):
    """Prints all the given matches

    Arguments:
        matches {list or np.ndarray(cv2.Dmatch)} -- List or array of cv2.Dmatch objects
    """

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


def draw_matches(img1, img2, matches):
    """Displays the two given images and their matches

    Arguments:
        img1 {np.ndarray} -- Left image to be displayed
        img2 {np.ndarray} -- Right image to be displayed
        matches {[type]} -- Common Points
    """

    img3 = cv2.drawMatches(img1.img, img1.keypoints, img2.img, img2.keypoints,
                           matches, outImg=np.array([]))
    plt.imshow(img3[:, :, [2, 1, 0]])
