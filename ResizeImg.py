from PIL import Image
import os

def resize_images(image_dir, resized_dir, size=(640,  640)):
    """
    Resizes images in the given directory to the specified size.

    Parameters:
    - image_dir (str): Path to the directory containing the images.
    - resized_dir (str): Path to the directory where resized images will be saved.
    - size (tuple): The target size for the images as (width, height).
    """
    # Create the resized directory if it doesn't exist
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)

    # Loop through each image in the directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # Add more extensions if needed
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            
            # Resize the image
            resized_img = img.resize(size, Image.LANCZOS)
            
            # Save resized image
            resized_img.save(os.path.join(resized_dir, filename))

    print(f"Images resized successfully to {size}.")

if __name__ == "__main__":
    image_dir = r"C:\Users\Kygo\Downloads\shoes.v4-allclasses_areshoes-original_raw-images.yolov7pytorch\train\images"

    resized_dir = r"c:\Users\Kygo\Desktop\frames\shoe"

    resize_images(image_dir, resized_dir)
