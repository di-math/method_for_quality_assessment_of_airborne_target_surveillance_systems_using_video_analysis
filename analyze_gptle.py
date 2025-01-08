import argparse
import cv2
import numpy as np

from lib.ground_plane_target_location_error.gptle_from_video import gptle_from_video

"""
This code defines the entry for a CLI program to analyse GPTD.
"""

# Define all ArUco markers, that are in the calibration patter by their id, the absolute position of their center, and their width (size)
CALIBRATION_PATTERN = {1: {"pos": {"x": -800, "y": -800}, "width": 1000},
                       2: {"pos": {"x": -800, "y": 800}, "width": 1000},
                       3: {"pos": {"x": 800, "y": 800}, "width": 1000},
                       4: {"pos": {"x": 800, "y": -800}, "width": 1000}}


def main():
    parser = argparse.ArgumentParser(
        prog='Video Target Track',
        description='Tracks the camera center movement relative to a calibration pattern.')
    parser.add_argument('path_to_input_video')
    parser.add_argument('-tx')
    parser.add_argument('-ty')
    parser.add_argument('-m', default=0)
    args = parser.parse_args()

    if args.tx and args.ty:
        target_coordinates = np.array([int(args.tx), int(args.ty)])
    else:
        target_coordinates = None

    gptle_from_video(args.path_to_input_video, cv2.aruco.DICT_ARUCO_ORIGINAL, CALIBRATION_PATTERN, target_coordinates=target_coordinates, output_mode=int(args.m))


if __name__ == '__main__':
    main()
