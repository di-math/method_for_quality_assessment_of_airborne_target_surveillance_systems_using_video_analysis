import os.path
import cv2 as cv
import numpy as np
import csv
from matplotlib import pyplot as plt

from lib.lens_distortion_correction.find_homography_with_distortion import find_homography_with_distortion, create_distortion_map, create_undistortion_map

"""
This snippet reproduces the experiment from the thesis in section 7.3
"""

def aruco_calculate_global_coordinates(ids, ids_global_measurements, corners):
    """
    Auxiliary function for sorting and matching source and destination points by the ids of detected ArUco markers
    """
    obj_points = []
    corners_all = []

    for i, id in enumerate(ids):
        try:
            center_x = ids_global_measurements[id[0]]["pos"]["x"]
            center_y = ids_global_measurements[id[0]]["pos"]["y"]
            marker_width = ids_global_measurements[id[0]]["width"]
        except KeyError:
            print(f"A marker with id {id[0]} was detected but is not defined in 'global_measurements' reference!")
            continue
        top_left = np.array([center_x - marker_width / 2, center_y + marker_width / 2], np.float32)
        top_right = np.array([center_x + marker_width / 2, center_y + marker_width / 2], np.float32)
        bottom_right = np.array([center_x + marker_width / 2, center_y - marker_width / 2], np.float32)
        bottom_left = np.array([center_x - marker_width / 2, center_y - marker_width / 2], np.float32)
        obj_points.append(top_left)
        obj_points.append(top_right)
        obj_points.append(bottom_right)
        obj_points.append(bottom_left)
        for corner in corners[i][0]:
            corners_all.append(corner)
    return np.array(corners_all, np.float32), np.array(obj_points, np.float32)

# Define all ArUco markers, that are in the calibration patter by their id, the absolute position of their center, and their width (size)
CALIBRATION_PATTERN = {1: {"pos": {"x": -800, "y": -800}, "width": 1000},
                       2: {"pos": {"x": -800, "y": 800}, "width": 1000},
                       3: {"pos": {"x": 800, "y": 800}, "width": 1000},
                       4: {"pos": {"x": 800, "y": -800}, "width": 1000}}

# Define test image input
input_images = "test/input/24equidistant_views/"
number_of_images = 24

# To store reprojection errors over all views
reprojection_errors = []

# Initialize ArUco detector
detector_params = cv.aruco.DetectorParameters()
detector_params.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
aruco_detector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_ARUCO_ORIGINAL), detector_params)

# Iterate all views
for i in range (1, number_of_images):
    image = cv.imread(os.path.join(input_images, f"view{i}.png"))

    # Distortion parameters
    c_x = (image.shape[1] / 2) - 30
    c_y = (image.shape[0] / 2) - 30
    k_1 = 20.0
    p_1 = 2.0
    p_2 = 0.0
    s_1 = 0.0
    s_2 = 0.0

    # Distort the image
    map_x = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_y = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_x, map_y = create_distortion_map(map_x, map_y, c_x, c_y, k_1, p_1, p_2, s_1, s_2)
    map_x, map_y = cv.convertMaps(map_x, map_y, cv.CV_32FC1)
    distorted_image = cv.remap(image, map_y, map_x, interpolation=(cv.INTERSECT_NONE | cv.WARP_RELATIVE_MAP))
    distorted_image_rgb = cv.cvtColor(distorted_image, cv.COLOR_BGR2RGB)

    # Uncomment to view each distorted image on the go
    # plt.imshow(distorted_image_rgb)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

    # Detect ArUco markers in the distorted image
    (corners, ids, rejected) = aruco_detector.detectMarkers(distorted_image_rgb)
    corners_all, object_points = aruco_calculate_global_coordinates(ids, CALIBRATION_PATTERN, corners)

    # Apply the method for estimating a distortion model and homography
    H, distortion_params, error = find_homography_with_distortion(
        object_points, corners_all, image.shape[1], image.shape[0])

    # Store the reprojection error for post-analysis
    reprojection_errors.append(error)

    # Undistort the image again using the estimated model
    map_x_undistord = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_y_undistord = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    map_x_undistord, map_y_undistord = create_undistortion_map(map_x_undistord, map_y_undistord, distortion_params[0], distortion_params[1], distortion_params[2], distortion_params[3], distortion_params[4], distortion_params[5], distortion_params[6])
    map_x_undistord , map_y_undistord  = cv.convertMaps(map_x_undistord , map_y_undistord , cv.CV_32FC1)
    undistored_image = cv.remap(distorted_image, map_y_undistord , map_x_undistord , interpolation=(cv.INTERSECT_NONE | cv.WARP_RELATIVE_MAP))
    undistored_image_rgb = cv.cvtColor(undistored_image, cv.COLOR_BGR2RGB)

    # Uncomment to view each undistorted image on the go
    # plt.imshow(undistored_image_rgb)
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()

plt.plot(reprojection_errors)
plt.title("Errors")
plt.show()

# Uncomment to store reprojections errors for all views in a .csv file
# output_file = "distortion_fly_by_data.csv"
# with open(output_file, mode='w', newline='') as file:
#     writer = csv.writer(file, delimiter=';')
#
#     writer.writerow(["nr", "value"])
#
#     for index, value in enumerate(reprojection_errors):
#         writer.writerow([index, value])
#
# print(f"Data has been written to {output_file}")

