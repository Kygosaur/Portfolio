import cv2
import numpy as np

def preprocess_video(input_video_path, output_video_path):
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise IOError("Could not open video file or camera.")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Loop through video frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Image Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            enhanced_frame = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

            # Noise Reduction
            denoised_frame = cv2.fastNlMeansDenoisingColored(enhanced_frame, None, 10, 10, 7, 21)

            out.write(denoised_frame)

        # Release resources
        cap.release()
        out.release()

    except Exception as e:
        print(f"Error processing video: {e}")
        # Ensure resources are released in case of an error
        if 'cap' in locals() or 'cap' in globals():
            cap.release()
        if 'out' in locals() or 'out' in globals():
            out.release()

if __name__ == "__main__":
    input_video_path = r"c:\Users\Kygo\Desktop\20231102\W20 Topside Pre-Fabrication Area-1 20231102 0925-0930.mp4"
    output_video_path = r"c:\Users\Kygo\Desktop\20231102\W20 Topside Pre-Fabrication Area-1 20231102 0925-0930_new.mp4"
    preprocess_video(input_video_path, output_video_path)