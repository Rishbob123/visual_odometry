import cv2
import os

# Path to folder with images
image_folder = r'C:\COSMOS\visual_odometry\dataset'
output_video = 'output_video.mp4'
fps = 5

# Get list of image files sorted by name
images = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])

# Read the first image to get dimensions
first_frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = first_frame.shape

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# Write each image frame
for image in images:
    frame = cv2.imread(os.path.join(image_folder, image))
    out.write(frame)

out.release()
print(f"Video saved as {output_video}")
