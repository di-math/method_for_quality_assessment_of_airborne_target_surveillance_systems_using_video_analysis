import numpy as np
import numpy.typing as npt


def precondition_points(points: npt.NDArray) -> tuple:
    """
    :param points: Array of 2D-source-points to be normalized by isotropic scaling.
    :return: Tuple (normalized_points, translation_matrix) - array of normalized 2D-points and translation applied to input points for normalization.
    """
    # Check the structure and number of points of the input array
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise AttributeError("points is not of shape (N, 2).")

    # Calculate the centroid of the given points
    centroid = np.mean(points, axis=0)

    # Calculate the average distance of all points to the centroid
    avg_dist_centroid = np.mean(
        list(np.linalg.norm(point - centroid) for point in points))
    s = np.sqrt(2) / avg_dist_centroid

    T = np.array([[s, 0, -centroid[0] * s],
                  [0, s, -centroid[1] * s], [0, 0, 1]])

    points_transformed = np.array(list(np.matmul(
        T, np.array([point[0], point[1], 1]))[:-1] for point in points))
    return points_transformed, T


def transform_points_homography(points: npt.NDArray, H: npt.NDArray) -> npt.NDArray:
    """
    Transforms points x by x'=Hx

    :param points: Array of 2D-source-points to be transformed.
    :param H: Homography.
    :return: 2d points transformed, x'
    """
    # Check the structure and number of points of the input array
    if len(points.shape) != 2 or points.shape[1] != 2:
        raise AttributeError("points is not of shape (N, 2).")
    if H.shape != (3, 3):
        raise AttributeError("Homography H is not of shape (3, 3).")

    points_transformed = []

    for i in range(len(points)):
        projected_point_hom = np.matmul(H, np.array(
            [points[i][0], points[i][1], 1]))
        projected_point_euc = np.array(
            [projected_point_hom[0] / projected_point_hom[2], projected_point_hom[1] / projected_point_hom[2]])
        points_transformed.append(projected_point_euc)

    return np.array(points_transformed)


def calculate_homography_lms(src_points: npt.NDArray, dest_points: npt.NDArray) -> npt.NDArray:
    """
    Calculate the homography between source and destination points.

    :param src_points: Array of 2D-source-points.
    :param dest_points: Array of 2D-destination-points.
    :return: Homography between source and destination points, using SVD.
    """
    # Check the structure and number of points of the two input arrays
    if len(src_points.shape) != 2 or src_points.shape[1] != 2:
        raise AttributeError("src_points is not of shape (N, 2).")
    if len(dest_points.shape) != 2 or dest_points.shape[1] != 2:
        raise AttributeError("dest_points is not of shape (N, 2).")
    if src_points.shape[0] != dest_points.shape[0]:
        raise Exception(f"src_points and dest_points do not contain the same number of points! src_points contains "
                        f"{src_points.shape[0]} points while dest_points contains {dest_points.shape[0]} points.")

    preconditioned_src_points, precondition_src_points_transformation_matrix = precondition_points(
        src_points)
    preconditioned_dest_points, precondition_dest_points_transformation_matrix = precondition_points(
        dest_points)

    A = np.zeros((len(preconditioned_src_points) * 2, 9),
                 dtype=np.float32)

    for i in range(len(preconditioned_src_points)):
        A[i * 2] = [-1.0 * preconditioned_src_points[i][0], -1.0 * preconditioned_src_points[i][1], -1.0, 0.0, 0.0, 0.0, preconditioned_dest_points[i]
        [0] * preconditioned_src_points[i][0], preconditioned_dest_points[i][0] * preconditioned_src_points[i][1], preconditioned_dest_points[i][0]]
        A[i * 2 + 1] = [0.0, 0.0, 0.0, -1.0 * preconditioned_src_points[i][0], -1.0 * preconditioned_src_points[i][1], -1.0, preconditioned_dest_points[i]
        [1] * preconditioned_src_points[i][0], preconditioned_dest_points[i][1] * preconditioned_src_points[i][1], preconditioned_dest_points[i][1]]

    U, Sigma, V_transpose = np.linalg.svd(A)
    H_tilde = np.reshape(V_transpose[-1], (3, 3))

    H = np.linalg.inv(
        precondition_dest_points_transformation_matrix) @ H_tilde @ precondition_src_points_transformation_matrix

    return H
