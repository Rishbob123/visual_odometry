import numpy as np
import cv2 as cv
import os
import math
from matplotlib import pyplot as plt

# === Load grayscale images ===
image_folder = r'C:\COSMOS\visual_odometry\dataset'
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
images = [cv.imread(os.path.join(image_folder, f), cv.IMREAD_GRAYSCALE) for f in image_files]
images = [img for img in images if img is not None]
assert len(images) > 1, "Not enough valid images found."

# === CARLA camera intrinsics ===
image_width = 1600
image_height = 900
fov_deg = 70

fov_rad = math.radians(fov_deg)
focal_length = image_width / (2 * math.tan(fov_rad / 2))

cx, cy = image_width / 2, image_height / 2

K = np.array([
    [focal_length, 0, cx],
    [0, focal_length, cy],
    [0, 0, 1]
])

# === Feature tracking using SIFT + FLANN ===
def track_features(img1, img2):
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return [], []

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
    pts4d /= pts4d[3]
    return pts4d

# === Scale estimation ===
def compute_scale(prev_pts, curr_pts):
    # Take only 3D coordinates
    prev = prev_pts[:3]
    curr = curr_pts[:3]

    # Trim to same number of points
    min_pts = min(prev.shape[1], curr.shape[1])
    prev = prev[:, :min_pts]
    curr = curr[:, :min_pts]

    num = np.linalg.norm(prev - np.roll(prev, 1, axis=1), axis=0)
    denom = np.linalg.norm(curr - np.roll(curr, 1, axis=1), axis=0) + 1e-8
    scale_factors = num / denom
    return np.median(scale_factors)


# === Initialize pose ===
R_total = np.eye(3)
t_total = np.zeros((3, 1))
trajectory_x, trajectory_z = [], []

# Initialize Kalman filter
x = np.array([[0], [0], [0], [0]])  # x, z, vx, vz
P = np.eye(4) * 100
F = np.array([[1, 0, 1, 0],
              [0, 1, 0, 1],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
R_kf = np.eye(2) * 5  # renamed to R_kf
Q = np.eye(4) * 0.01

# === Start with first frame pair ===
pts1, pts2 = track_features(images[0], images[1])
if len(pts1) < 8:
    raise ValueError("Not enough feature matches in first frame.")

E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC, 0.999, 0.4)
pts1, pts2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]

_, R, t, _ = cv.recoverPose(E, pts1, pts2, K)
prev_cloud = triangulate(np.eye(3), np.zeros((3, 1)), pts1, pts2, K)

t_total += R_total @ t
R_total = R_total @ R

# Predict and update Kalman filter
x_pred = F @ x
P_pred = F @ P @ F.T + Q
z = np.array([[t_total[0][0]], [t_total[2][0]]])
y = z - H @ x_pred
S = H @ P_pred @ H.T + R_kf
K_gain = P_pred @ H.T @ np.linalg.inv(S)
x = x_pred + K_gain @ y
P = (np.eye(4) - K_gain @ H) @ P_pred

trajectory_x.append(x[0][0])
trajectory_z.append(x[1][0])

# === Loop through remaining frame pairs ===
for i in range(1, len(images) - 1):
    pts1, pts2 = track_features(images[i], images[i + 1])
    if len(pts1) < 8:
        continue

    E, mask = cv.findEssentialMat(pts2, pts1, K, cv.RANSAC, 0.999, 0.4)
    pts1, pts2 = pts1[mask.ravel() == 1], pts2[mask.ravel() == 1]

    if len(pts1) < 8:
        continue

    _, R, t, _ = cv.recoverPose(E, pts1, pts2, K)

    curr_cloud = triangulate(R, t, pts1, pts2, K)
    s = compute_scale(prev_cloud, curr_cloud)
    prev_cloud = curr_cloud

    t_total += s * R_total @ t
    R_total = R_total @ R

    # Predict and update Kalman filter
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q
    z = np.array([[-t_total[0][0]], [-t_total[2][0]]])
    y = z - H @ x_pred
    S = H @ P_pred @ H.T + R_kf
    K_gain = P_pred @ H.T @ np.linalg.inv(S)
    x = x_pred + K_gain @ y
    P = (np.eye(4) - K_gain @ H) @ P_pred

    trajectory_x.append(x[0][0])
    trajectory_z.append(x[1][0])

# === Plot trajectory ===
plt.plot(trajectory_x, trajectory_z, color='green', label='Estimated Trajectory')
plt.xlabel("x (meters)")
plt.ylabel("z (meters)")
plt.title("2D-2D Feature Matching KF Video 1")
plt.legend()
plt.grid(True)
plt.show()
