# This script is modified from https://github.com/RAMcCracken/CS5199_Breast_Cancer_Detection_Project/blob/main/src/data_preprocessing/ddsm_crop.py
# Modified by Brandon Linnett on 24/07/2024

import os
import cv2
import imutils
from imutils import contours

def main():
    base_dir = "/data/bl70/validate/ProcessedImages/CBIS-DDSM"
    segmented_base_dir = os.path.join(base_dir, "Segmented")
    print(f"Starting processing in base directory: {base_dir}")
    
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path) and dir_name != "Segmented":
            print(f"\nProcessing directory: {dir_name}")
            process_directory(dir_path, segmented_base_dir)

def process_directory(dir_path, segmented_base_dir):
    dir_name = os.path.basename(dir_path)
    segmented_dir = os.path.join(segmented_base_dir, f"Segmented{dir_name}")
    os.makedirs(segmented_dir, exist_ok=True)
    print(f"Created segmented directory: {segmented_dir}")
    
    image_count = 0
    processed_count = 0
    skipped_count = 0
    for filename in os.listdir(dir_path):
        if filename.endswith(('.png', '.PNG')):
            image_count += 1
            image_path = os.path.join(dir_path, filename)
            output_path = os.path.join(segmented_dir, filename)
            
            if os.path.exists(output_path):
                print(f"Skipping image {image_count}: {filename} (already processed)")
                skipped_count += 1
                continue
            
            print(f"Processing image {image_count}: {filename}")
            process_image(image_path, segmented_dir)
            processed_count += 1
    
    print(f"Processed {processed_count} images in {dir_name}")
    print(f"Skipped {skipped_count} already processed images in {dir_name}")
    print(f"Total images: {image_count}")

def process_image(image_path, segmented_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    is_left = check_is_left(image_path)
    print(f"Image orientation: {'Left' if is_left else 'Right'}")
    cropped = segment_image(image, is_left)
    
    save_processed_image(cropped, image_path, segmented_dir)

def save_processed_image(image, original_path, segmented_dir):
    file_name = os.path.basename(original_path)
    new_path = os.path.join(segmented_dir, file_name)
    cv2.imwrite(new_path, image)
    print(f"Saved segmented image: {new_path}")

def check_is_left(image_path):
    if "LEFT" in image_path:
        return True
    elif "RIGHT" in image_path:
        return False
    else:
        print(f"Could not determine left or right for image: {image_path}")
        return None

def segment_image(image, is_left):
    print("Starting image segmentation")
    border_y = image.shape[0]//16
    border_x = image.shape[1]//16
    y = border_y
    h = image.shape[0] - border_y
    if is_left:
        x = 10
        w = image.shape[1] - (border_x)
    else:
        x = border_x
        w = image.shape[1] - 10
    
    img = image[y:h, x:w]
    
    print("Applying image processing techniques")
    shifted = cv2.pyrMeanShiftFiltering(img, 21, 51)
    blur = cv2.blur(shifted, (5,5))
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    print("Finding contours")
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if len(cnts) == 0:
        print("No contours found. Returning original image.")
        return img

    try:
        (cnts, _) = contours.sort_contours(cnts)
        largest_contour = max(cnts, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(largest_contour)
        
        border_y = image.shape[0]//25
        border_x = image.shape[1]//25
        
        Y1 = max(y - border_y, 0)
        Y2 = min(y + h + border_y, image.shape[0])
        X2 = min(x + w + border_x, image.shape[1])
        
        crop = img[Y1:Y2, x:X2]
        print("Image segmentation completed")
        return crop
    except ValueError as e:
        print(f"Error during contour processing: {e}")
        print("Returning original image.")
        return img

if __name__ == '__main__':
    print("Starting script execution")
    main()
    print("Script execution completed")