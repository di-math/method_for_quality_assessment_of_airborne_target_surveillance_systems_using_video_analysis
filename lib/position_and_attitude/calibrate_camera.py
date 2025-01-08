import cv2 as cv
import numpy as np


def calibrate_camera(source_points, destination_points, u_0, v_0):
    # Calculate homographies for each view received
    homographies = []
    for i, elem in enumerate(source_points):
        H, _ = cv.findHomography(source_points[i], destination_points[i])
        homographies.append(H)

    V = []
    b = []

    for H in homographies:
        h1 = H[:, 0]
        h2 = H[:, 1]
        h1x, h1y, h1z = h1
        h2x, h2y, h2z = h2

        # Orthogonality constraint (h1^T * ω * h2 = 0), Simplified orthogonality constraint using known (u_0, v_0) and gamma=0
        V.append([
            h1x * h2x - u_0 * (h1x * h2z + h1z * h2x) +
            h1z * h2z * u_0**2,  # Coefficient of B11
            h1y * h2y - v_0 * (h1y * h2z + h1z * h2y) + \
            h1z * h2z * v_0**2   # Coefficient of B22
        ])

        b.append(-h1z * h2z)

        # Equality of norms constraint
        V.append([
            h1x**2 - h2x**2 - 2 * u_0 *
                (h1x * h1z - h2x * h2z) + h1z**2 * u_0**2 -
            h2z**2 * u_0**2,  # Coefficient of B11
            h1y**2 - h2y**2 - 2 * v_0 * \
                (h1y * h1z - h2y * h2z) + h1z**2 * v_0**2 - \
            h2z**2 * v_0**2   # Coefficient of B22
        ])
        b.append(-(h1z**2 - h2z**2))

    V = np.array(V)
    b = np.array(b)

    # Solve the linear system V * [B11, B22].T = -b using least squares
    b = np.linalg.lstsq(V, b, rcond=None)[0]

    # Extract B11 and B22
    B11, B22 = b[:2]

    # Calculate focal lengths from B11 and B22
    f_x = np.sqrt(1 / B11)
    f_y = np.sqrt(1 / B22)

    # Principal point coordinates are known: u_0 and v_0
    c_x = u_0
    c_y = v_0

    K = np.array([[f_x, 0, u_0],
                 [0, f_y, v_0],
                 [0, 0, 1]])

    rvecs = []
    tvecs = []
    for H in homographies:
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

    return K, np.array(rvecs), np.array(tvecs)
