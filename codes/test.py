import cv2

# Option 1: raw string
image = cv2.imread(r"C:\COSMOS\visual_odometry\dataset\Town01_type001_subtype0001_scenario00003_001.jpg")

# Option 2: double backslashes
# image = cv2.imread("C:\\COSMOS\\visual_odometry\\dataset\\Town01_type001_subtype0001_scenario00003_001.jpg")

height, width = image.shape[:2]
print(f"Width: {width}, Height: {height}")
