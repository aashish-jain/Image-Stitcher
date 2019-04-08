def cv_matcher(img1, desc1, img2, desc2, kp1=None, kp2=None, threshold=0.75):
    brute_force_matcher = cv2.BFMatcher()

    # Match descriptors.
    matches = brute_force_matcher.knnMatch(desc1, desc2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < threshold * n.distance:
            good.append(m)

    if kp1 != None and kp2 != None:
        img3 = cv2.drawMatches(img1, kp1, img2, kp2,
                               good, outImg=np.array([]))  # , matchColor=(0, 255, 0))
        show_image(img3)

def cv2_homography(matches, keypoints1, keypoints2):    
    pts1 = np.array([keypoints1[m.queryIdx].pt for m in matches])
    pts2 = np.array([keypoints2[m.trainIdx].pt for m in matches])
    return cv2.findHomography(pts1, pts2, cv2.RANSAC)[0]