import argparse
import cv2
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from enum import Enum


class GptleDetectorOutputMode(Enum):
    PATH = 0
    FRAME_GPTLE = 1


def aruco_calculate_global_coordinates(ids, ids_global_measurements):
    obj_points = []
    for id in ids:
        center_x = ids_global_measurements[id[0]]["pos"]["x"]
        center_y = ids_global_measurements[id[0]]["pos"]["y"]
        marker_width = ids_global_measurements[id[0]]["width"]
        top_left = np.array([center_x - marker_width / 2, center_y + marker_width / 2], np.float32)
        top_right = np.array([center_x + marker_width / 2, center_y + marker_width / 2], np.float32)
        bottom_right = np.array([center_x + marker_width / 2, center_y - marker_width / 2], np.float32)
        bottom_left = np.array([center_x - marker_width / 2, center_y - marker_width / 2], np.float32)
        obj_points.append(top_left)
        obj_points.append(top_right)
        obj_points.append(bottom_right)
        obj_points.append(bottom_left)
    return np.array(obj_points, np.float32)


def gptle_from_video(path_to_input_video: str, aruco_dict: int, calibration_pattern_definition: dict, output_path="./output", target_coordinates=None, output_mode=GptleDetectorOutputMode.PATH) -> None:
    """
    TODO -currently only .mkv supported
    :param output_mode:
    :param calibration_pattern_definition:
    :param path_to_input_video:
    :param aruco_dict:
    :param output_path:
    :param target_coordinates:
    :return:
    """

    # Check if video file exists
    if not os.path.isfile(path_to_input_video):
        raise FileNotFoundError(f"File {path_to_input_video} not found.")
    # Check if file has the right type
    if os.path.splitext(path_to_input_video)[-1].lower() != ".mkv":
        raise Exception(f"File {path_to_input_video} is not of extension .mkv.")
    # Create output directory
    current_time = datetime.datetime.now()
    output_path = os.path.join(output_path,
                               f"video_track_{current_time.hour}_{current_time.minute}_{current_time.second}_{current_time.year}_{current_time.month}_{current_time.day}")
    try:
        os.makedirs(output_path)
    except:
        raise Exception(f"Could not create output directory {output_path}")
    # Check if the provided target coordinates are of the right shape
    if target_coordinates is not None and len(target_coordinates) != 2:
        raise Exception(f"Provided target_coordinates are of length {len(target_coordinates)} but should be of length 2 (2d point in calibration plane coordinates)")
    # Prepare aruco detector
    detector_params = cv2.aruco.DetectorParameters()
    detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    aruco_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(aruco_dict), detector_params)
    # Prepare video capture
    video_capture = cv2.VideoCapture(path_to_input_video)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    fps = float(video_capture.get(cv2.CAP_PROP_FPS))
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(output_path, 'video_analyzed.mkv'), fourcc, fps, (frame_width, frame_height), True)

    sum_gptle = 0
    gptles = []
    los_ground_intersection_points = []
    frame_nr = 0

    while video_capture.isOpened():
        # Retrieve new frame
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive a new frame. Probably the last frame has been reached. Exiting reading the stream.")
            break
        # Detect Aruco-Markers in the frame
        (corners, ids, rejected) = aruco_detector.detectMarkers(frame)
        # Calculate the object-points in global coordinates, so their array indices are the same as their image point correspondences
        object_points = aruco_calculate_global_coordinates(ids, calibration_pattern_definition)
        # Reformat the detected corners to an image_points array
        image_points = []
        for corner_i in corners:
            for corner_k in corner_i[0]:
                image_points.append(corner_k)

        # Estimate homography between image and object points
        H = cv2.findHomography(np.array(image_points), object_points)[0]
        H_inv = np.linalg.inv(H)
        # Check if a homography could be estimated, if not write an empty frame, else proceed below
        if H.shape != (3, 3):
            print(f"No Homography could be estimated for frame {frame_nr}, it will be skipped.")
            gptles.append(None)
            los_ground_intersection_points.append(None)
            out.write(frame)
            frame_nr = frame_nr + 1
            continue
        else:
            # Back-project the image center to retrieve los intersection with the ground
            los_intersect = np.matmul(H, np.array([frame_width / 2, frame_height / 2, 1]))
            # Normalize
            los_intersect_x = los_intersect[0] / los_intersect[2]
            los_intersect_y = los_intersect[1] / los_intersect[2]
            los_ground_intersection_points.append(np.array((los_intersect_x, los_intersect_y)))
            # Calculate the gptle
            # If no target has been provided, this first los intersection with the ground plane will be that target for all further frames
            if target_coordinates is None:
                target_coordinates = np.array((los_intersect_x, los_intersect_y))
            gptle = np.array((los_intersect_x - target_coordinates[0], los_intersect_y - target_coordinates[0]))
            gptles.append(gptle)
            sum_gptle = sum_gptle + math.sqrt(gptle[0] ** 2 + gptle[1] ** 2)

            # If output_mode PATH draw a path of all intersections from all frames, else just visualize the gptle
            if output_mode == 0:
                for center_point in los_ground_intersection_points:
                    points_to_draw = [
                        (center_point[0] - 1, center_point[1] + 1),
                        (center_point[0], center_point[1] + 1),
                        (center_point[0] + 1, center_point[1] + 1),
                        (center_point[0] - 1, center_point[1]),
                        center_point,
                        (center_point[0] + 1, center_point[1]),
                        (center_point[0] - 1, center_point[1] - 1),
                        (center_point[0], center_point[1] - 1),
                        (center_point[0] + 1, center_point[1] - 1)
                    ]

                    points_to_draw_translated = []

                    for point in points_to_draw:
                        point_translated = np.matmul(H_inv, np.array([point[0], point[1], 1]))
                        point_translated_norm_x = point_translated[0] / point_translated[2]
                        point_translated_norm_y = point_translated[1] / point_translated[2]
                        points_to_draw_translated.append((point_translated_norm_x, point_translated_norm_y))

                    for point in points_to_draw_translated:
                        frame[math.floor(point[1])][math.floor(point[0])] = np.array([255, 0, 0])
            else:
                # Calculate the image coordinates of the target and draw a line accordingly
                target_coordinates_image_homog = np.matmul(H_inv, np.array([target_coordinates[0], target_coordinates[1], 1]))
                target_coordinates_image = np.array([target_coordinates_image_homog[0]/target_coordinates_image_homog[2], target_coordinates_image_homog[1]/target_coordinates_image_homog[2]])
                cv2.line(frame, (int(frame_width / 2), int(frame_height / 2)), (int(target_coordinates_image[0]), int(target_coordinates_image[1])), (255, 0, 0), 2)

            out.write(frame)

    # Release video-capture and -output
    video_capture.release()
    out.release()
