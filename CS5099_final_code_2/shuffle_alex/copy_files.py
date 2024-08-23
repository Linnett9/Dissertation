import os
import shutil

# Base directory and new directory for Final
base_dir = "/raid/bl70/Final/Training"
new_dir = "/raid/bl70/Final2/Training2"
variations_to_keep = [
    "CL2.0_TG8x8_WN0.01",
]

# Augmentations to consider
augmentations_to_keep = [
    "Original_CL",
    "Original_Flipped_Horizontally",
    "Rotated_90_degrees_CL",
    "Translated_10_x_-10_y_CL"
]

def copy_files(src_dir, dest_dir, variations, augmentations):
    total_copied = 0

    for aug in augmentations:
        print(f"\nCopying files for augmentation: {aug}")
        copied_count = 0

        for root, _, files in os.walk(src_dir):
            for file in files:
                # Ensure the file is one of the variations we want to keep
                if any(variation in file for variation in variations):
                    # Ensure we only copy the desired augmentations
                    if aug in file:
                        # Skip files that have both augmentations (e.g., rotated and flipped)
                        if "Flipped_Horizontally" in file and aug == "Rotated_90_degrees.png":
                            continue
                        if "Rotated_90_degrees" in file and aug == "_Original_Flipped_Horizontally.png":
                            continue

                        file_path = os.path.join(root, file)
                        rel_path = os.path.relpath(file_path, src_dir)
                        dest_path = os.path.join(dest_dir, rel_path)
                        
                        # Create destination directory if it doesn't exist
                        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                        
                        # Copy the file
                        shutil.copy2(file_path, dest_path)
                        copied_count += 1
                        
                        # Print the first 5 files copied
                        if copied_count <= 5:
                            print(f"Copied: {dest_path}")

        print(f"Total copied {copied_count} files for augmentation {aug}")
        total_copied += copied_count
    
    return total_copied

# Copy files for Benign and Malignant directories
if __name__ == "__main__":
    benign_src_dir = os.path.join(base_dir, "Benign")
    malignant_src_dir = os.path.join(base_dir, "Malignant")
    
    benign_dest_dir = os.path.join(new_dir, "Benign")
    malignant_dest_dir = os.path.join(new_dir, "Malignant")
    
    # Copying files from Benign directory
    benign_copied = copy_files(benign_src_dir, benign_dest_dir, variations_to_keep, augmentations_to_keep)
    print(f"\nTotal images copied to Benign: {benign_copied}")
    
    # Copying files from Malignant directory
    malignant_copied = copy_files(malignant_src_dir, malignant_dest_dir, variations_to_keep, augmentations_to_keep)
    print(f"\nTotal images copied to Malignant: {malignant_copied}")
    
    print(f"\nCopying completed. Total images copied: {benign_copied + malignant_copied}")
