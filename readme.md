# Method for Quality Assessment of Airborne Target Surveillance using Video Analysis
Accompanying code repository for the thesis.

## Determining Ground Plane Target Location Error
use `analyze_gptle.py`, e.g.:
```commandline
$ python analze_gptle.py "path/to/test/input/video01.mkv [-tx <target x location> -ty <target y location> -m <mode>]"
```
Modes:
1. Draw GPTLE path over all frames (trace)
2. Draw each individual GPTLE vector in the coresponding frame

## Lens Distortion Correction
To test with synthetic data, execute:
```commandline
$ python test/test_homography_with_distortion_synthetic_data.py
```

## Position and Attitude from Video

