# Method for Quality Assessment of Airborne Target Surveillance using Video Analysis
Accompanying code repository for the thesis.

## Determining Ground Plane Target Location Error
use `analyze_gptle.py`, e.g.:
```commandline
$ python analyze_gptle.py "path/to/test/input/video.mkv" [-tx <target x location> -ty <target y location> -m <mode>]
```
Modes:
1. Draw GPTLE path over all frames (trace)
2. Draw each individual GPTLE vector in the coresponding frame

To reproduce the experiment from the thesis in section 7.2, run:
```commandline
$ python analyze_gptle.py "test/input/video03_experiment.mkv"
```

## Lens Distortion Correction
To test with synthetic data, execute:
```commandline
$ python test/test_homography_with_distortion_synthetic_data.py
```

To reproduce experiment from the thesis in section 7.1, run:
```commandline
$ python image_undistortion_using_aruco_calibration_pattern.py
```

## Position and Attitude from Video
To reproduce experiment from the thesis in section 7.3, run:
```commandline
$ python position_from_video.py "test\input\video04_experiment.mkv"
```