import numpy as np

# From lecture slides - CSE573


def get_homography_matrix(pts1, pts2):
    """Returns a 3*3 homograpy matrix that maps pts1 to pts2. CAlculates the 
    homography matrix by using the using the eigen value of the smallest eigen vector 
    of set of equations formed by given points or Direct Linear Transform.

    Arguments:
        pts1 {np.ndarray} -- List of points for which homography is computed
        pts2 {np.ndarray} -- List of points using which homography is computed

    Returns:
        np.ndarray -- A matrix mapping pts1 to pts2
    """

    eqn_list = []
    # Form set of equations from the given points
    for (x1, y1), (x2, y2) in zip(pts1, pts2):
        row1 = [x1, y1, 1, 0, 0, 0, -x1*x2, -y1*x2, -x2]
        row2 = [0, 0, 0, x1, y1, 1, -x1*y2, -y1*y2, -y2]
        eqn_list.append(row1)
        eqn_list.append(row2)

    # Solve for A.h = 0 i.e., h = eigen value corresponding to smallest eigen value of A'A
    A = np.array(eqn_list)
    eig_values, eig_vectors = np.linalg.eig(np.dot(A.T, A))
    h_index = eig_values.argmin()
    h = eig_vectors[:, h_index]

    # Convert h to a unit vector
    h = h / np.linalg.norm(h)
    h /= h[-1]
    h = h.reshape(3, 3)
    return h
