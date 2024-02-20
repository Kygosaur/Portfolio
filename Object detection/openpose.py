import cv2
import numpy as np
from openpose import pyopenpose as op

# Set up OpenPose parameters
params = dict()
params["model_folder"] = "/path/to/openpose/models/"

# Start OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Process webcam input
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Perform pose estimation
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Display the results
    cv2.imshow("OpenPose", datum.cvOutputData)
    if cv2.waitKey(1) &  0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
