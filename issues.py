import os

def check_annotation_files(directory):
    """
    Checks if each .jpg file in the specified directory has a corresponding .txt file.
    Prints a message only for .jpg files without a corresponding .txt file.

    Parameters:
    - directory (str): The path to the directory containing the .jpg files.
    """
    
    files = os.listdir(directory)
    
    # Filter out .jpg files
    jpg_files = [f for f in files if f.endswith('.jpg')]
    
    for jpg_file in jpg_files:
        # expected .txt file based on jpg
        txt_file = os.path.splitext(jpg_file)[0] + '.txt'
        
        # Check if the .txt file exists
        if txt_file not in files:
            print(f"No .txt file found for {jpg_file}.")

if __name__ == "__main__":
    directory = r"C:\Users\Kygo\Portfolio\YOLOv7\yolov7\val"  # Update this to your directory
    check_annotation_files(directory)
