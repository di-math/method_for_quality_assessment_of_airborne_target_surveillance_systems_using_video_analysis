import numpy as np
import cv2
import matplotlib.pyplot as plt

from lib.lens_distortion_correction.find_homography_with_distortion import distort_points, find_homography_with_distortion, undistort_points

image_width = 1920
image_height = 1080
point_size = 5

image = np.zeros((image_height, image_width), dtype=np.uint8)

rows = 12
cols = 18
x_spacing = image_width // (cols + 1)
y_spacing = image_height // (rows + 1)

src_points = []
for row in range(1, rows + 1):
    for col in range(1, cols + 1):
        x = col * x_spacing
        y = row * y_spacing
        src_points.append([x, y])
src_points = np.array(src_points, dtype=np.float32)

for point in src_points:
    cv2.circle(image, (int(point[0]), int(point[1])), point_size, 255, -1)

plt.imshow(image, cmap='gray')
plt.title("Original Image with undistorted Points")
plt.show()

cx, cy = image_width / 2 - 6, image_height / 2 - 38
k1, p1, p2, s1, s2 = -3.7, 0.002, 0.005, 0.002, 0.001
distorted_points = distort_points(dist_params=(
    cx, cy, k1, p1, p2, s1, s2), undistorted_points=src_points)

distorted_image = np.zeros((image_height, image_width), dtype=np.uint8)
for point in distorted_points:
    cv2.circle(distorted_image, (int(point[0]), int(point[1])), point_size, 255, -1)

plt.imshow(distorted_image, cmap='gray')
plt.title("Distorted Image with distorted Points")
plt.show()

H, distortion_params, error = find_homography_with_distortion(
    src_points, distorted_points, image_width, image_height)

print("Estimated Homography:\n", H)
print("Estimated Distortion Parameters:", distortion_params)
print("Reprojection Error:", error)

undistored_points = undistort_points(distortion_params, distorted_points)

opt_test_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
for point in src_points:
    cv2.circle(opt_test_image, (int(point[0]), int(point[1])), point_size, (0, 255, 0), -1)
for point in undistored_points:
    cv2.circle(opt_test_image,
               (int(point[0]), int(point[1])), point_size, (255, 255, 255), -1)

plt.imshow(opt_test_image)
plt.title("Final Image")
plt.show()
