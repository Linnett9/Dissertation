import os
import pydicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import wiener
from scipy.ndimage import gaussian_filter
from skimage import img_as_float, img_as_uint, img_as_ubyte
import time

def verify_path(path):
    if os.path.exists(path):
        print(f"Path verified: {path}", flush=True)
    else:
        print(f"Path does not exist: {path}", flush=True)
        print("Available directories:", os.listdir(os.path.dirname(path)), flush=True)
        exit(1)

def load_dicom_images(path, start_idx=0, batch_size=10):
    print(f"Loading DICOM images from: {path}, start index: {start_idx}", flush=True)
    images = []
    patient_ids = []
    file_names = []
    total_files = [os.path.join(root, file) for root, _, files in os.walk(path) for file in files if file.endswith('.dcm')]

    for filepath in total_files[start_idx:start_idx + batch_size]:
        try:
            dicom = pydicom.dcmread(filepath)
            print(f"Successfully read DICOM file: {filepath}", flush=True)
            
            if not hasattr(dicom, 'PatientID'):
                raise ValueError(f"Missing PatientID in file {filepath}")
            if 'PixelData' not in dicom:
                raise ValueError(f"Missing PixelData in file {filepath}")
                
            image = dicom.pixel_array
            print(f"Image shape: {image.shape}, dtype: {image.dtype}", flush=True)
            
            if image.dtype != np.uint16:
                image = image.astype(np.uint16)
            image = image.astype(np.float32)  # Convert to float32
            images.append(image)
            patient_ids.append(dicom.PatientID)
            file_names.append(os.path.basename(filepath))
            
        except (pydicom.errors.InvalidDicomError, AttributeError, KeyError, FileNotFoundError, ValueError) as e:
            print(f"Skipping file {filepath} due to error: {e}", flush=True)
            continue

    print(f"Loaded {len(images)} images from {path}", flush=True)
    return images, patient_ids, file_names

def apply_wiener(image, noise=None):
    try:
        print(f"Applying Wiener filter...", flush=True)
        image = img_as_float(image)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))  # Scale to [0, 1]
        image -= 0.5  # Center to [-0.5, 0.5]
        smoothed_image = gaussian_filter(image, sigma=1)
        if noise is not None:
            noise = max(noise, 1e-10)  # Avoid zero noise
            filtered_image = wiener(smoothed_image, noise=noise)
        else:
            filtered_image = wiener(smoothed_image)

        # Clip any invalid values
        filtered_image = np.clip(filtered_image, 0, 1)
        filtered_image = img_as_uint(filtered_image)
        return filtered_image

    except Exception as e:
        print(f"Error applying Wiener filter: {e}", flush=True)
        return image

def normalize_image(image):
    try:
        print(f"Normalizing image...", flush=True)
        norm_image = (image - np.min(image)) * (65535.0 / (np.max(image) - np.min(image)))
        return norm_image.astype(np.uint16)
    except Exception as e:
        print(f"Error normalizing image: {e}", flush=True)
        return image

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    try:
        print(f"Applying CLAHE...", flush=True)
        if image.dtype != np.uint16:
            image = np.uint16(image * 65535)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply(image)
        return result
    except Exception as e:
        print(f"Error applying CLAHE: {e}", flush=True)
        return image

def save_images(images, output_dir, patient_id, original_file_name):
    try:
        print(f"Saving images...", flush=True)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        for i, image in enumerate(images):
            # Strip the .dcm extension if it exists
            base_name = os.path.splitext(original_file_name)[0]  # Removes the .dcm extension
            # Check for invalid values before saving
            if np.isnan(image).any() or np.isinf(image).any():
                raise ValueError(f"Image contains NaN or infinity values")

            output_path = os.path.join(output_dir, f"{patient_id}_{base_name}.png")  # Ensure saving as PNG
            plt.imsave(output_path, image, cmap='gray')  # Save the image as a grayscale PNG
            print(f"Saved: {output_path}", flush=True)
    except Exception as e:
        print(f"Error saving images: {e}", flush=True)


def load_processed_images_list(processed_list_path):
    if os.path.exists(processed_list_path):
        with open(processed_list_path, 'r') as file:
            processed_images = file.read().splitlines()
    else:
        processed_images = []
    return set(processed_images)

def save_processed_image(image_name, processed_list_path):
    try:
        with open(processed_list_path, 'a') as file:
            file.write(image_name + "\n")
    except Exception as e:
        print(f"Error saving processed image name: {e}", flush=True)

def process_image(image, patient_id, original_file_name, dataset_name, noise_levels, clip_limits, tile_grid_sizes, output_dir_base, processed_list_path):
    start_time = time.time()

    if processed_list_path and original_file_name in processed_images:
        print(f"Image {original_file_name} has already been processed. Skipping...", flush=True)
        return

    for clip_limit in clip_limits:
        for tile_grid_size in tile_grid_sizes:
            try:
                clahe_applied = apply_clahe(normalize_image(image), clip_limit=clip_limit, tile_grid_size=tile_grid_size)
                for noise in noise_levels:
                    wiener_filtered = apply_wiener(clahe_applied, noise=noise)
                    normalized_wiener = normalize_image(wiener_filtered)
                    parameters = f"CL{clip_limit}_TG{tile_grid_size[0]}x{tile_grid_size[1]}_WN{noise}"
                    output_dir = os.path.join(output_dir_base, dataset_name, parameters)
                    save_images([normalized_wiener], output_dir, patient_id, original_file_name)
            except Exception as e:
                print(f"Error processing image {original_file_name}: {e}", flush=True)
                continue

    if processed_list_path:
        save_processed_image(original_file_name, processed_list_path)

    end_time = time.time()
    print(f"Finished processing image {original_file_name}. Time taken: {end_time - start_time} seconds", flush=True)

def process_dataset(path, dataset_name, noise_levels, clip_limits, tile_grid_sizes, output_dir_base, processed_list_path, batch_size=10):
    start_idx = 0
    while True:
        images, patient_ids, file_names = load_dicom_images(path, start_idx=start_idx, batch_size=batch_size)
        if not images:
            break
        for image, patient_id, file_name in zip(images, patient_ids, file_names):
            process_image(image, patient_id, file_name, dataset_name, noise_levels, clip_limits, tile_grid_sizes, output_dir_base, processed_list_path)
        start_idx += batch_size

def main():
    #cbis_ddsm_path = "/data/bl70/CBIS-DDSM/CBIS-DDSM"
    cmmd_path = "/data/bl70/CMMD/CMMD"
  #  verify_path(cbis_ddsm_path)
    verify_path(cmmd_path)
    noise_levels = [0.01, 0.1, 1.0]
    clip_limits = [2.0, 3.0, 5.0]
    tile_grid_sizes = [(8, 8), (16, 16)]

    
    #host_output_dir_base = os.path.expanduser("/data/bl70/CBIS-DDSM/CBIS-DDSM/processed")
    host_output_dir_base = os.path.expanduser("/data/bl70/CMMD/CMMD/processed")
    #processed_list_path = os.path.expanduser("/data/bl70/CBIS-DDSM/CBIS-DDSM/processed/processed.txt")
    processed_list_path = os.path.expanduser("/data/bl70/CMMD/CMMD/processed/processed.txt")
    global processed_images
    processed_images = load_processed_images_list(processed_list_path)

   # process_dataset(cbis_ddsm_path, "CBIS-DDSM", noise_levels, clip_limits, tile_grid_sizes, host_output_dir_base, processed_list_path, batch_size=10)
    process_dataset(cmmd_path, "CMMD", noise_levels, clip_limits, tile_grid_sizes, host_output_dir_base, processed_list_path, batch_size=10)
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script terminated by user.", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
