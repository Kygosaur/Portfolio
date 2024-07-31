import os
import glob
import shutil

def process_unpaired_files(directory):
    # Create output directories
    output_dir = os.path.join(directory, "unpaired_files_output")
    unpaired_images_dir = os.path.join(output_dir, "unpaired_images")
    unpaired_txt_dir = os.path.join(output_dir, "unpaired_txt")
    os.makedirs(unpaired_images_dir, exist_ok=True)
    os.makedirs(unpaired_txt_dir, exist_ok=True)

    # Get all image files (assuming .jpg, .png, and .jpeg extensions)
    image_files = glob.glob(os.path.join(directory, "*.jpg")) + \
                  glob.glob(os.path.join(directory, "*.png")) + \
                  glob.glob(os.path.join(directory, "*.jpeg"))
    
    # Get all .txt files
    txt_files = glob.glob(os.path.join(directory, "*.txt"))
    
    # Convert to dictionaries of basenames (filename without extension) to full paths
    image_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
    txt_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in txt_files}
    
    # Find unpaired files
    images_without_txt = set(image_dict.keys()) - set(txt_dict.keys())
    txt_without_images = set(txt_dict.keys()) - set(image_dict.keys())
    
    copied_images = 0
    copied_txt = 0

    # Prepare summary information
    summary = []
    summary.append(f"Total image files: {len(image_files)}")
    summary.append(f"Total .txt files: {len(txt_files)}")
    
    # Process unpaired images
    if images_without_txt:
        summary.append("\nImages without corresponding .txt files:")
        for basename in sorted(images_without_txt):
            src = image_dict[basename]
            dst = os.path.join(unpaired_images_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied_images += 1
            summary.append(f"  {basename}")

    # Process unpaired .txt files
    if txt_without_images:
        summary.append("\nText files without corresponding images:")
        for basename in sorted(txt_without_images):
            src = txt_dict[basename]
            dst = os.path.join(unpaired_txt_dir, os.path.basename(src))
            shutil.copy2(src, dst)
            copied_txt += 1
            summary.append(f"  {basename}")

    # Add copy summary
    summary.append(f"\nCopied {copied_images} unpaired image files to: {unpaired_images_dir}")
    summary.append(f"Copied {copied_txt} unpaired .txt files to: {unpaired_txt_dir}")

    # Write summary to file
    summary_file_path = os.path.join(output_dir, "unpaired_files_summary.txt")
    with open(summary_file_path, 'w') as f:
        f.write('\n'.join(summary))

    # Print summary to console
    print('\n'.join(summary))
    print(f"\nSummary written to: {summary_file_path}")

# Example usage
directory = r"c:\Users\jack\Desktop\awa"
process_unpaired_files(directory)