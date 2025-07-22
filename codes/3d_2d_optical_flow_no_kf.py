import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import math

# ==== PARAMETERS ====
dataset_path = r'C:\COSMOS\visual_odometry\dataset'

# CARLA camera intrinsics
image_width = 1600
image_height = 900
fov_deg = 70

fov_rad = math.radians(fov_deg)
focal_length = image_width / (2 * math.tan(fov_rad / 2))

cx, cy = image_width / 2, image_height / 2

K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# ==== HELPER FUNCTIONS ====
def load_images_from_folder(folder):
    image_paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    images = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in image_paths]
    return images

def get_good_matches(p0, p1, status):
    good_old = p0[status == 1]
    good_new = p1[status == 1]
    return good_old, good_new

# ==== MAIN ====
images = load_images_from_folder(dataset_path)
assert len(images) > 1, "Need at least 2 images"

traj = np.zeros((600, 600, 3), dtype=np.uint8)

cur_R = np.eye(3)
cur_t = np.zeros((3, 1))

trajectory = []

# Detect initial features
feature_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=8)
p0 = cv2.goodFeaturesToTrack(images[0], mask=None, **feature_params)

for i in range(1, len(images)):
    img1, img2 = images[i - 1], images[i]

    # Track features
    p1, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None)

    # Filter good points
    good_p0, good_p1 = get_good_matches(p0, p1, status)

    # Find Essential Matrix
    E, mask = cv2.findEssentialMat(good_p1, good_p0, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    _, R, t, mask_pose = cv2.recoverPose(E, good_p1, good_p0, K)

    # Update pose
    cur_t += cur_R @ t
    cur_R = R @ cur_R

    trajectory.append((cur_t[0][0], cur_t[2][0]))

    # Redetect features if too few
    if len(good_p1) < 200:
        p0 = cv2.goodFeaturesToTrack(img2, mask=None, **feature_params)
    else:
        p0 = good_p1.reshape(-1, 1, 2)

# ==== PLOT ====
xs, zs = zip(*trajectory)
plt.figure()
plt.plot(xs, zs, color='green', label='Estimated trajectory')
plt.xlabel('x')
plt.ylabel('z')
plt.title('3D-2D Optical Flow No KF Video 1')
plt.legend()
plt.grid(True)
plt.show()
