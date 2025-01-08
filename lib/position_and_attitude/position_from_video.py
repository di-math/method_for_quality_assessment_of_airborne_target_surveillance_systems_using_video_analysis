import argparse
import cv2
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os.path
from calibrate_camera import calibrate_camera

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


def main():
    parser = argparse.ArgumentParser(
        prog='Video Camera Track',
        description='Tracks the camera center movement relative to a calibration pattern on a given plane.')
    parser.add_argument('path_to_input_video')
    args = parser.parse_args()
    position_from_video(args.path_to_input_video, cv2.aruco.DICT_ARUCO_ORIGINAL)


def position_from_video(path_to_input_video: str, aruco_dict: int, output_path="./output") -> None:
    # Check if file exists
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

    # Iterate frames
    source_points = []
    destination_points = []

    while video_capture.isOpened():
        # Retrieve new frame
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive a new frame. Exiting reading the stream.")
            break
        # Detect Aruco-Markers in the frame
        (corners, ids, rejected) = aruco_detector.detectMarkers(frame)
        # Calculate an array with the object-points in global coordinates - TODO: make this a dynamic definition
        object_points = aruco_calculate_global_coordinates(ids, {1: {"pos": {"x": -800, "y": -800}, "width": 1000},
                       2: {"pos": {"x": -800, "y": 800}, "width": 1000},
                       3: {"pos": {"x": 800, "y": 800}, "width": 1000},
                       4: {"pos": {"x": 800, "y": -800}, "width": 1000}})
        # Reformat the detected corners-array - TODO: this can be done better
        corners_all = []
        for corner_i in corners:
            for corner_k in corner_i[0]:
                corners_all.append(corner_k)
        
        source_points.append(object_points)
        destination_points.append(corners_all)

    video_capture.release()
    K, rvecs, tvecs = calibrate_camera(np.array(source_points), np.array(destination_points), frame_width/2, frame_height/2)
    video_capture = cv2.VideoCapture(path_to_input_video)
    
    frame_cnt = 0
    while video_capture.isOpened():
        # Retrieve new frame
        ret, frame = video_capture.read()
        if not ret:
            print("Can't receive a new frame. Exiting reading the stream.")
            break
        
        object_points = np.array([[0, 0, 0], [1000, 0, 0], [0, 1000, 0], [0, 0, 1000]], np.float32)
        dist_coeff = np.array([[0], [0], [0], [0], [0]], np.float32)
        image_points, _ = cv2.projectPoints(object_points, rvecs[frame_cnt], tvecs[frame_cnt], K, dist_coeff)

        cv2.line(frame,(int(image_points[0][0][0]), int(image_points[0][0][1])),(int(image_points[1][0][0]), int(image_points[1][0][1])),(0,0,255),5)
        cv2.line(frame,(int(image_points[0][0][0]), int(image_points[0][0][1])),(int(image_points[2][0][0]), int(image_points[2][0][1])),(0,255,0),5)
        cv2.line(frame,(int(image_points[0][0][0]), int(image_points[0][0][1])),(int(image_points[3][0][0]), int(image_points[3][0][1])),(255,0,0),5)

        out.write(frame)
        frame_cnt = frame_cnt + 1

    # Release video-capture and -output
    video_capture.release()
    out.release()


if __name__ == '__main__':
    main()
