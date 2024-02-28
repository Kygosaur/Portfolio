from PIL import Image
import os

def adjust_yolo_annotations(image_dir, resized_dir, size=(640,   640)):
    """
    Adjusts the YOLO annotations of images in the given directory to match the resized images.

    Parameters:
    - image_dir (str): Path to the directory containing the original images and their annotations.
    - resized_dir (str): Path to the directory containing the resized images.
    - size (tuple): The target size for the images as (width, height).
    """
    # Ensure the resized_dir exists
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)

    print("Adjusting YOLO annotations and resizing images...")
    # Loop through each image file in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Add more extensions if needed
            # Check if a .txt file exists for the image
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_path = os.path.join(image_dir, txt_filename)
            if not os.path.exists(txt_path):
                print(f"No annotation found for {filename}. Skipping.")
                continue

            # Resize the image
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            original_size = img.size  # Get the actual original size of the image
            if original_size != (608,  608):  # Check if the original size matches the expected size
                print(f"Warning: Original size of {filename} does not match the expected size. Skipping annotation adjustment.")
                continue
            resized_img = img.resize(size, Image.LANCZOS)
            resized_img_path = os.path.join(resized_dir, filename)
            resized_img.save(resized_img_path)

            # Calculate the scaling factors based on the actual original size
            scale_x = size[0] / original_size[0]
            scale_y = size[1] / original_size[1]

            with open(txt_path, 'r') as file:
                lines = file.readlines()

            resized_txt_path = os.path.join(resized_dir, txt_filename)
            with open(resized_txt_path, 'w') as file:
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) ==   5:
                        class_id, center_x, center_y, width, height = parts
                        # Scale the coordinates
                        center_x = float(center_x) * scale_x
                        center_y = float(center_y) * scale_y
                        width = float(width) * scale_x
                        height = float(height) * scale_y
                        # Write the adjusted annotation
                        file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

if __name__ == "__main__":
    image_dir = r"C:\Users\Kygo\Downloads\shoes.v4-allclasses_areshoes-original_raw-images.yolov7pytorch\train\images"
    resized_dir = r"c:\Users\Kygo\Desktop\frames\test"
    adjust_yolo_annotations(image_dir, resized_dir)
