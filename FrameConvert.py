import cv2
import os

video_path = r"C:\Users\Kygo\Desktop\2 NOV\W20 Topside Pre-Fabrication Area-1 20231102 0910-0915.mp4"
output_dir = "frames"

specific_folder = "datasets"

# Create full output path
full_output_path = os.path.join(output_dir, specific_folder)

# Create specific folder if it doesn't exist
if not os.path.exists(full_output_path):
    os.makedirs(full_output_path)

# Open the video capture object
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Frame count
frame_count = 0

while True:
    ret, frame = cap.read()

    # Check if frame is read correctly
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # Save frame as image
    filename = os.path.join(full_output_path, f"frame_{frame_count}.jpg")
    cv2.imwrite(filename, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()

print(f"Extracted {frame_count} frames to {full_output_path}")
