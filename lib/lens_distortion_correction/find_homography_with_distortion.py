import cv2
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy

from lib.homography.calculate_homography import calculate_homography_lms, transform_points_homography

# For scaling the distortion parameters
sigma_k_1 = 1e-8
sigma_p_1 = 1e-4
sigma_p_2 = 1e-4
sigma_s_1 = 1e-3
sigma_s_2 = 1e-3


def find_homography_with_distortion(src_points: npt.NDArray, dest_points: npt.NDArray, image_width: int, image_height: int, epsilon_threshold=0.01) -> tuple:
    """
    Calculates the homography and the distortion parameters between the src and the destination points.

    :param src_points: Array of 2D-source-points.
    :param dest_points: Array of 2D-destination-points.
    :param image_width: Width of the image buffer plane.
    :param image_height: Height of the image buffer plane.
    :param epsilon_threshold: optional, epsilon threshold for initial distortion parameter estimation.
    :return: Tuple (Homography, Distortion Coefficients, Reprojection Error)
    """
    # Check the structure and number of points of the two input arrays
    if len(src_points.shape) != 2 or src_points.shape[1] != 2:
        raise AttributeError("src_points is not of shape (N, 2).")
    if len(dest_points.shape) != 2 or dest_points.shape[1] != 2:
        raise AttributeError("dest_points is not of shape (N, 2).")
    if src_points.shape[0] != dest_points.shape[0]:
        raise Exception(f"src_points and dest_points do not contain the same number of points! src_points contains "
                        f"{src_points.shape[0]} points while dest_points contains {dest_points.shape[0]} points.")

    # Compute the initial homography H
    H = calculate_homography_lms(src_points, dest_points)

    # Compute initial cost
    F_0 = mean_squared_transfer_error(transform_points_homography(src_points, H), dest_points)

    # Implementation and formula indices referring to (Gao and Yin, 2012) Computing a complete camera lens distortion model by planar homography
    c_x = image_width / 2
    c_y = image_height / 2

    points_opt = np.zeros((len(src_points), 2))
    first_iter = True
    while True:
        # Approximate ideal image points using homography
        approx_ideal_image_points = np.zeros((len(src_points), 2))
        for i, point in enumerate(src_points):
            approx_ideal_image_point = H @ np.array([point[0], point[1], 1])
            approx_ideal_image_points[i] = np.array(
                [approx_ideal_image_point[0] / approx_ideal_image_point[2], approx_ideal_image_point[1] / approx_ideal_image_point[2]])

        # Set up matrix A and B, by 16, 17
        A = np.zeros((len(src_points) * 2, 3), dtype=np.float64)
        B = np.zeros((len(src_points) * 2, 1), dtype=np.float64)
        for i in range(len(approx_ideal_image_points)):
            x_u = approx_ideal_image_points[i][0] - c_x
            y_u = approx_ideal_image_points[i][1] - c_y
            if first_iter:
                x_d = dest_points[i][0] - c_x
                y_d = dest_points[i][1] - c_y
            else:
                x_d = points_opt[i][0] - c_x
                y_d = points_opt[i][1] - c_y

            r_2 = x_u ** 2 + y_u ** 2
            A[i * 2] = [sigma_k_1 * x_u * r_2, sigma_p_1 * (3 * (x_u ** 2) + y_u ** 2), sigma_p_2 * 2 * x_u * y_u]
            A[i * 2 + 1] = [sigma_k_1 * y_u * r_2, sigma_p_1 * 2 * x_u * y_u, sigma_p_2 * ((x_u ** 2) + 3 * (y_u ** 2))]
            B[i * 2] = [x_d - x_u]
            B[i * 2 + 1] = [y_d - y_u]

        # Calculate m_s by 18
        m_s = np.linalg.inv(np.linalg.matrix_transpose(A) @ A) @ np.linalg.matrix_transpose(A) @ B

        k_1 = m_s[0][0]
        p_1 = m_s[1][0]
        p_2 = m_s[2][0]

        # Iteratively compute x_opt, y_opt by 19
        if first_iter:
            points_to_iterate = dest_points
            first_iter = False
        else:
            points_to_iterate = points_opt

        for i, point in enumerate(points_to_iterate):
            x_opt = point[0] - c_x
            y_opt = point[1] - c_y
            for k in range(1):
                r_d2 = (x_opt ** 2) + (y_opt ** 2)
                d_x = sigma_p_1 * p_1 * (3 * (x_opt ** 2) + y_opt ** 2) + sigma_p_2 * 2 * p_2 * x_opt * y_opt
                d_y = sigma_p_1 * 2 * p_1 * x_opt * y_opt + sigma_p_2 * p_2 * (3 * (y_opt ** 2) + x_opt ** 2)
                x_opt = (x_opt - d_x) / (1 + (k_1 * sigma_k_1) * r_d2)
                y_opt = (y_opt - d_y) / (1 + (k_1 * sigma_k_1) * r_d2)
            points_opt[i] = np.array([x_opt + c_x, y_opt + c_y])

        # Compute new cost
        F_1 = mean_squared_transfer_error(transform_points_homography(src_points, H), points_opt)

        if abs((F_0 - F_1) / F_1) < epsilon_threshold:
            break
        else:
            # Compute the new homography H
            H = calculate_homography_lms(src_points, points_opt)
            F_0 = F_1
            continue  # just to be explicit

    # Set up matrix A and B, by 21
    A = np.zeros((len(src_points) * 2, 5), dtype=np.float64)
    B = np.zeros((len(src_points) * 2, 1), dtype=np.float64)
    for i in range(len(points_opt)):
        x_u = points_opt[i][0] - c_x
        y_u = points_opt[i][1] - c_y
        x_d = dest_points[i][0] - c_x
        y_d = dest_points[i][1] - c_y
        r_2 = x_d ** 2 + y_d ** 2
        A[i * 2] = [sigma_k_1 * x_d * r_2, sigma_p_1 * (3 * (x_d ** 2) + (y_d ** 2)), sigma_p_2 * 2 * x_d * y_d, sigma_s_1 * r_2, 0]
        A[i * 2 + 1] = [sigma_k_1 * y_d * r_2, sigma_p_1 * 2 * x_d * y_d, sigma_p_2 * ((x_d ** 2) + (3 * (y_d ** 2))), 0, sigma_s_2 * r_2]
        B[i * 2] = [x_d - x_u]
        B[i * 2 + 1] = [y_d - y_u]

    m_sl = np.linalg.inv(np.linalg.matrix_transpose(A) @ A) @ np.linalg.matrix_transpose(A) @ B

    m_linear = (c_x, c_y, m_sl[0][0], m_sl[1][0], m_sl[2][0], m_sl[3][0], m_sl[4][0])

    def cost_function(params, src_points, dest_points):
        dist_parameters = params[:7]
        H = params[7:].reshape(3, 3)
        distorted_points_opt = undistort_points(dist_parameters, dest_points)
        source_points_transformed = transform_points_homography(src_points, H)
        residuals = distorted_points_opt - source_points_transformed
        return residuals.flatten()

    optimized_params  = scipy.optimize.least_squares(cost_function, np.concatenate([np.array(m_linear), H.flatten()]), args=(src_points, dest_points), method='lm').x
    m_nonlinear = optimized_params[:7]
    H = optimized_params[7:].reshape(3, 3)

    F_nonlinear = mean_squared_transfer_error(undistort_points(m_nonlinear, dest_points), transform_points_homography(src_points, H))

    F_linear = mean_squared_transfer_error(undistort_points(m_linear, dest_points), transform_points_homography(src_points, H))

    if F_nonlinear <= F_linear:
        print(f"Chose nonlinear solution: F_linear={F_linear} and F_nonlinear={F_nonlinear}")
        return H, m_nonlinear, F_nonlinear
    else:
        print(f"Chose linear solution: F_linear={F_linear} and F_nonlinear={F_nonlinear}")
        return H, m_linear, F_linear


def mean_squared_transfer_error(points_a: npt.NDArray, points_b: npt.NDArray) -> float:
    """
    :param points_a: Array of 2D-points.
    :param points_b: Array of 2D-points.
    :return: Mean squared transfer error between points_a and points_b.
    """
    # Check the structure and number of points of the two input arrays and of the homography H
    if len(points_a.shape) != 2 or points_a.shape[1] != 2:
        raise AttributeError("src_points is not of shape (N, 2).")
    if len(points_b.shape) != 2 or points_b.shape[1] != 2:
        raise AttributeError("dest_points is not of shape (N, 2).")
    if points_a.shape[0] != points_b.shape[0]:
        raise Exception(f"src_points and dest_points do not contain the same number of points! src_points contains "
                        f"{points_a.shape[0]} points while dest_points contains {points_b.shape[0]} points.")

    sum_euclidean_distances = 0

    for i in range(len(points_a)):
        sum_euclidean_distances = sum_euclidean_distances + \
                                  ((points_a[i][0] - points_b[i][0]) ** 2 +
                                   (points_a[i][1] - points_b[i][1]) ** 2)
    F = (1 / points_a.shape[0]) * sum_euclidean_distances

    return F


def undistort_points(dist_params: tuple, distorted_points: npt.NDArray) -> npt.NDArray:
    """
    :param dist_params: Tuple of distortion parameters (c_x, c_y, k_1, p_1, p_2, s_1, s_2).
    :param distorted_points: Array of distorted 2D-image-points.
    :return: Undistorted points
    """
    # Check the structure and number of points of the two inputs
    if len(distorted_points.shape) != 2 or distorted_points.shape[1] != 2:
        raise AttributeError("distorted_points is not of shape (N, 2).")
    if len(dist_params) != 7:
        raise AttributeError("dist_params is not of length 7.")

    undistorted_points = np.copy(distorted_points)
    c_x, c_y, k_1, p_1, p_2, s_1, s_2 = dist_params

    for i in range(len(undistorted_points)):
        x_u = undistorted_points[i][0] - c_x
        y_u = undistorted_points[i][1] - c_y
        r_2 = x_u ** 2 + y_u ** 2

        D_x = (k_1 * sigma_k_1) * x_u * r_2 + (p_1 * sigma_p_1) * (3 * x_u ** 2 + y_u ** 2) + 2 * (p_2 * sigma_p_2) * x_u * y_u + (s_1 * sigma_s_1) * r_2
        D_y = (k_1 * sigma_k_1) * y_u * r_2 + (p_2 * sigma_p_2) * (x_u ** 2 + 3 * y_u ** 2) + 2 * (p_1 * sigma_p_1) * x_u * y_u + (s_2 * sigma_s_2) * r_2

        undistorted_points[i][0] = undistorted_points[i][0] - D_x
        undistorted_points[i][1] = undistorted_points[i][1] - D_y

    return undistorted_points


def distort_points(dist_params: tuple, undistorted_points: npt.NDArray) -> npt.NDArray:
    """
    :param dist_params: Tuple of distortion parameters (c_x, c_y, k_1, p_1, p_2, s_1, s_2).
    :param undistorted_points: Array of undistorted 2D-image-points.
    :return: distorted points
    """
    # Check the structure and number of points of the two inputs
    if len(undistorted_points.shape) != 2 or undistorted_points.shape[1] != 2:
        raise AttributeError("undistorted_points is not of shape (N, 2).")
    if len(dist_params) != 7:
        raise AttributeError("dist_params is not of length 7.")

    distorted_points = np.copy(undistorted_points)
    c_x, c_y, k_1, p_1, p_2, s_1, s_2 = dist_params

    for i in range(len(distorted_points)):
        x_u = distorted_points[i][0] - c_x
        y_u = distorted_points[i][1] - c_y
        r_2 = x_u ** 2 + y_u ** 2

        D_x = (k_1 * sigma_k_1) * x_u * r_2 + (p_1 * sigma_p_1) * \
              (3 * x_u ** 2 + y_u ** 2) + 2 * (p_2 * sigma_p_2) * \
              x_u * y_u + (s_1 * sigma_s_1) * r_2
        D_y = (k_1 * sigma_k_1) * y_u * r_2 + (p_2 * sigma_p_2) * \
              (x_u ** 2 + 3 * y_u ** 2) + 2 * (p_1 * sigma_p_1) * \
              x_u * y_u + (s_2 * sigma_s_2) * r_2

        distorted_points[i][0] = distorted_points[i][0] + D_x
        distorted_points[i][1] = distorted_points[i][1] + D_y

    return distorted_points


def create_distortion_map(map_x, map_y, c_x, c_y, k_1, p_1, p_2, s_1, s_2):
    for y, row in enumerate(map_x):
        for x, pixel in enumerate(row):
            x_u = x - c_x
            y_u = y - c_y
            r_2 = x_u**2 + y_u **2
            map_x[y][x] = -1 * ((k_1 * sigma_k_1) * x_u * r_2 + (p_1 * sigma_p_1) * \
              (3 * x_u ** 2 + y_u ** 2) + 2 * (p_2 * sigma_p_2) * \
              x_u * y_u + (s_1 * sigma_s_1) * r_2)
    for y, row in enumerate(map_y):
        for x, pixel in enumerate(row):
            x_u = x - c_x
            y_u = y - c_y
            r_2 = x_u**2 + y_u **2
            map_x[y][x] = -1 * ((k_1 * sigma_k_1) * y_u * r_2 + (p_2 * sigma_p_2) * \
              (x_u ** 2 + 3 * y_u ** 2) + 2 * (p_1 * sigma_p_1) * \
              x_u * y_u + (s_2 * sigma_s_2) * r_2)
    return map_x, map_y


def create_undistortion_map(map_x, map_y, c_x, c_y, k_1, p_1, p_2, s_1, s_2):
    for y, row in enumerate(map_x):
        for x, pixel in enumerate(row):
            x_u = x - c_x
            y_u = y - c_y
            r_2 = x_u**2 + y_u **2
            map_x[y][x] = -1 * ((k_1 * sigma_k_1) * x_u * r_2 + (p_1 * sigma_p_1) * \
              (3 * x_u ** 2 + y_u ** 2) + 2 * (p_2 * sigma_p_2) * \
              x_u * y_u + (s_1 * sigma_s_1) * r_2)
    for y, row in enumerate(map_y):
        for x, pixel in enumerate(row):
            x_u = x - c_x
            y_u = y - c_y
            r_2 = x_u**2 + y_u **2
            map_x[y][x] = -1 * ((k_1 * sigma_k_1) * y_u * r_2 + (p_2 * sigma_p_2) * \
              (x_u ** 2 + 3 * y_u ** 2) + 2 * (p_1 * sigma_p_1) * \
              x_u * y_u + (s_2 * sigma_s_2) * r_2)
    return map_x, map_y