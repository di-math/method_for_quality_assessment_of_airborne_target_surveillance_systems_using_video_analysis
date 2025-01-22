import math

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def calibrate_camera_with_zoom(source_points, destination_points, u_0, v_0):
    # To store reprojection errors for each window for post-analysis
    reprojection_errors_analysis = []

    # To store the final rvecs, tvecs and Ks for each frame, or None if no stable estimation could be done
    final_rvecs = []
    final_tvecs = []
    final_Ks = []

    # Temp store for rvecs, tvecs and K of the "previous" window
    previous_window_rvecs = []
    previous_window_tvecs = []
    previous_window_K = None

    # Calculate homographies for each view received
    homographies = []
    for i, elem in enumerate(source_points):
        try:
            H, _ = cv.findHomography(source_points[i], destination_points[i])
        except:
            print(f"No Homography could be estimated for frame {i}, it will be skipped.")
            homographies.append(None)
            continue
        homographies.append(H)

    # Define threshold for the reprojection error (tau), initial window size, and initial pointer
    thresh_reprj_error = 15
    initial_window_size = 20

    window_size = initial_window_size
    pointer = 0

    while True:
        # Iteration anchor
        if pointer+window_size > len(homographies):
            final_rvecs.extend(previous_window_rvecs)
            final_tvecs.extend(previous_window_tvecs)
            for i in range(window_size-1):
                final_Ks.append(previous_window_K)
            break

        # Setup V and b for current window
        V = []
        b = []
        for H in homographies[pointer:pointer+window_size]:
            # Ignore if no homography could have been estimated for this view
            if H is None:
                print("H is none")
                continue
            h1 = H[:, 0]
            h2 = H[:, 1]
            h1x, h1y, h1z = h1
            h2x, h2y, h2z = h2

            # Orthogonality constraint (h1^T * ω * h2 = 0), simplified orthogonality constraint using known (u_0, v_0) and gamma=0
            V.append([
                h1x * h2x - u_0 * (h1x * h2z + h1z * h2x) +
                h1z * h2z * u_0 ** 2,  # Coefficient of B11
                h1y * h2y - v_0 * (h1y * h2z + h1z * h2y) + \
                h1z * h2z * v_0 ** 2  # Coefficient of B22
            ])

            b.append(-h1z * h2z)

            # Equality of norms constraint
            V.append([
                h1x ** 2 - h2x ** 2 - 2 * u_0 *
                (h1x * h1z - h2x * h2z) + h1z ** 2 * u_0 ** 2 -
                h2z ** 2 * u_0 ** 2,  # Coefficient of B11
                h1y ** 2 - h2y ** 2 - 2 * v_0 * \
                (h1y * h1z - h2y * h2z) + h1z ** 2 * v_0 ** 2 - \
                h2z ** 2 * v_0 ** 2  # Coefficient of B22
            ])
            b.append(-(h1z ** 2 - h2z ** 2))

        V = np.array(V)
        b = np.array(b)

        # Solve the linear system V * [B11, B22].T = -b using least-squares method
        x, residuals, rank, s = np.linalg.lstsq(V, b, rcond=None)
        cond = np.linalg.cond(V)

        # Extract B11 and B22
        B11, B22 = x[:2]

        # Calculate focal lengths from B11 and B22
        f_x = np.sqrt(1 / B11)
        f_y = np.sqrt(1 / B22)

        # Principal point coordinates are known: u_0 and v_0
        c_x = u_0
        c_y = v_0

        # Setup K for the current window
        K = np.array([[f_x, 0, u_0],
                      [0, f_y, v_0],
                      [0, 0, 1]])

        # Calculate rvecs and tvecs for the views in the current window
        rvecs = []
        tvecs = []
        for H in homographies[pointer:pointer+window_size]:
            if H is None:
                rvecs.append(None)
                tvecs.append(None)
                continue
            h1 = H[:, 0]
            h2 = H[:, 1]
            h3 = H[:, 2]
            lamb_da = np.linalg.norm(np.dot(np.linalg.inv(K), h1), 2)

            r1 = np.dot(np.linalg.inv(K), h1) / lamb_da
            r2 = np.dot(np.linalg.inv(K), h2) / lamb_da
            r3 = np.cross(r1, r2)
            R = np.column_stack((r1, r2, r3))
            rvec, _ = cv.Rodrigues(R)
            rvecs.append(rvec.reshape(3, 1))
            t = np.dot(np.linalg.inv(K), h3) / lamb_da
            tvecs.append(t.reshape(3, 1))

        # Calculate reprojection error for the current window
        reprojection_error = 0
        for i, H in enumerate(homographies[pointer:pointer+window_size]):
            if H is None:
                continue
            dist_coeff = np.array([[0], [0], [0], [0], [0]], np.float32)
            lis = []
            for elem in source_points[pointer+i]:
                tmp_elem = list(elem)
                tmp_elem.append(0)
                lis.append(tmp_elem)
            image_points_reprojected, _ = cv.projectPoints(np.array(lis, np.float32), rvecs[i], tvecs[i], K, dist_coeff)
            reprj_sum = 0
            for k, reprj_point in enumerate(image_points_reprojected):
                reprj_sum = reprj_sum + (reprj_point[0][0] - destination_points[pointer+i][k][0])**2 + (reprj_point[0][1] - destination_points[pointer+i][k][1])**2
            reprojection_error = reprojection_error + (reprj_sum / image_points_reprojected.shape[0])
        reprojection_error = reprojection_error / (V.shape[0] / 2)
        reprojection_errors_analysis.append(reprojection_error)

        # Decide based on reprojection error
        if reprojection_error > thresh_reprj_error or math.isnan(f_x) or math.isnan(f_y):
            print(f"Reprojection error is: {reprojection_error}")
            if window_size == initial_window_size:
                print(f"Window at {pointer} with window size {initial_window_size} is not stable. Moving window...")
                final_rvecs.append(None)
                final_tvecs.append(None)
                final_Ks.append(None)
                pointer = pointer + 1
            else:
                print(f"Window at {pointer} with window size {window_size} became unstable. Resetting window size and moving window...")
                final_rvecs.extend(previous_window_rvecs)
                final_tvecs.extend(previous_window_tvecs)
                for i in range(window_size-1):
                    final_Ks.append(previous_window_K)
                pointer = pointer + window_size - 1
                window_size = initial_window_size
        else:
            print(f"Window at {pointer} with window size {window_size} is stable. Extending window size by one...")
            previous_window_rvecs = rvecs
            previous_window_tvecs = tvecs
            previous_window_K = K
            window_size = window_size + 1

    # Uncomment to plot reprojection errors over all window iterations
    plt.plot(reprojection_errors_analysis)
    plt.title("Reprojection Errors")
    plt.show()

    return final_Ks, final_rvecs, final_tvecs