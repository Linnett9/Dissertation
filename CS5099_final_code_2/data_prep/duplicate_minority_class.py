import os
import shutil

# Define the base directory and target count
base_dir = "/data/bl70/CBIS-DDSM/Training/Malignant"
target_count = 84  # Target number of images to match the Majority class

# Ensure the base directory exists
if os.path.isdir(base_dir):
    # Get a list of all minority images
    malignant_images = [f for f in os.listdir(base_dir) if f.endswith('.png')]
    num_malignant_images = len(malignant_images)
    
    # Calculate how many additional images are needed
    num_needed_images = target_count - num_malignant_images
    
    # Ensure we only proceed if duplication is needed
    if num_needed_images > 0:
        full_duplications = num_needed_images // num_malignant_images
        partial_duplications = num_needed_images % num_malignant_images
        
        # Perform full duplications
        for i in range(full_duplications):
            for image_name in malignant_images:
                src_path = os.path.join(base_dir, image_name)
                new_image_name = f"{os.path.splitext(image_name)[0]}_dup{i+1}{os.path.splitext(image_name)[1]}"
                dst_path = os.path.join(base_dir, new_image_name)
                shutil.copy2(src_path, dst_path)
        
        # Perform partial duplication
        for i in range(partial_duplications):
            image_name = malignant_images[i]
            src_path = os.path.join(base_dir, image_name)
            new_image_name = f"{os.path.splitext(image_name)[0]}_dup_partial{os.path.splitext(image_name)[1]}"
            dst_path = os.path.join(base_dir, new_image_name)
            shutil.copy2(src_path, dst_path)
    
    print(f"Malignant directory now contains {len(os.listdir(base_dir))} images.")

else:
    print(f"Directory {base_dir} does not exist. Please check the path.")

print("Duplication of the minority class is complete.")