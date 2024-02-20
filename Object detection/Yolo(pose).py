from ultralytics import YOLO

# Load the pose estimation model
model = YOLO('yolov8m-pose.pt')

# Perform pose estimation on the webcam input
results = model(source=0, show=True, conf=0.3, save=False)

# The 'results' object contains the detection results
# You can access the keypoints and other information from the results
