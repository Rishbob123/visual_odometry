import numpy as np
import cv2 as cv
import os
import math
from matplotlib import pyplot as plt

# === Load grayscale images ===
image_folder = r'C:\COSMOS\visual_odometry\dataset'
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
images = [cv.imread(os.path.join(image_folder, f), cv.IMREAD_GRAYSCALE) for f in image_files]

# === Camera calibration matrix (example for KITTI) ===
image_width = 800   # replace with your actual image width
image_height = 600  # replace with your actual image height
fov_deg = 90        # default CARLA camera FOV

fov_rad = math.radians(fov_deg)
focal_length = image_width / (2 * math.tan(fov_rad / 2))

K = np.array([
    [focal_length, 0, image_width / 2],
    [0, focal_length, image_height / 2],
    [0, 0, 1]
])
# === Feature tracking using SIFT + FLANN ===
def track_features(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    flann = cv.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(np.float32(des1), np.float32(des2), k=2)

    pts1, pts2 = [], []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)

    return np.float32(pts1), np.float32(pts2)

# === Triangulation ===
def triangulate(R, t, pts1, pts2, K):
    proj1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
    proj2 = K @ np.hstack((R, t))
    pts4d = cv.triangulatePoints(proj1, proj2, pts1.T, pts2.T)
    return pts4d / pts4d[3]

# === Scale estimation ===
def compute_scale(prev_pts, curr_pts):
    prev = prev_pts[:3, :-1]
    curr = curr_pts[:3, :-1]
    scale_factors = np.linalg.norm(prev - np.roll(prev, 1, axis=1), axis=0) / \
                    (np.linalg.norm(curr - np.roll(curr, 1, axis=1), axis=0) + 1e-8)
    return np.median(scale_factors)

# === Initialize trajectory ===
R_total = np.eye(3)
t_total = np.zeros((3, 1))
trajectory_x, trajectory_z = [], []

# === Start with first pair ===
pts1, pts2 = track_features(images[0], images[1])
E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC, 0.999, 0.4)
pts1, pts2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]
_, R, t, _ = cv.recoverPose(E, pts1, pts2, K)
prev_cloud = triangulate(np.eye(3), np.zeros((3, 1)), pts1, pts2, K)
t_total += R_total @ t
R_total = R_total @ R
trajectory_x.append(t_total[0][0])
trajectory_z.append(t_total[2][0])

# === Loop through remaining frames ===
for i in range(1, len(images) - 1):
    pts1, pts2 = track_features(images[i], images[i + 1])
    E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC, 0.999, 0.4)
    pts1, pts2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]
    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

    curr_cloud = triangulate(R, t, pts1, pts2, K)
    s = compute_scale(prev_cloud, curr_cloud)
    prev_cloud = curr_cloud

    t_total += s * R_total @ t
    R_total = R_total @ R

    trajectory_x.append(t_total[0][0])
    trajectory_z.append(t_total[2][0])

# === Plot trajectory ===
plt.plot(trajectory_x, trajectory_z, color='green', label='Estimated Trajectory')
plt.xlabel("x (meters)")
plt.ylabel("z (meters)")
plt.title("Visual Odometry Trajectory")
plt.legend()
plt.grid(True)
plt.show()
