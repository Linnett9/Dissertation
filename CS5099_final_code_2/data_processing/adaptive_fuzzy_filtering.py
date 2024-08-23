import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median
from skimage.morphology import disk
from skimage.color import rgb2gray
from skimage import img_as_float, img_as_uint
import time

def adaptive_fuzzy_median(image, radius=1):
    footprint = disk(radius)
    filtered_image = median(image, footprint=footprint)
    return filtered_image

def apply_additional_filtering(input_dirs, output_base_dir, radius=1):
    total_start_time = time.time()
    for input_dir in input_dirs:
        output_dir = os.path.join(output_base_dir, os.path.basename(input_dir))
        os.makedirs(output_dir, exist_ok=True)
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    try:
                        # Load image
                        image = plt.imread(file_path)
                        # Check the number of dimensions
                        if image.ndim == 3:
                            # Convert RGB image to grayscale
                            image = rgb2gray(image)
                        elif image.ndim != 2:
                            raise ValueError(f"Image at {file_path} has incorrect number of dimensions: {image.ndim}")

                        # Convert image to float
                        image = img_as_float(image)

                        # Start timing
                        start_time = time.time()

                        # Apply adaptive fuzzy median filter
                        filtered_image = adaptive_fuzzy_median(image, radius=radius)

                        # Save the filtered image
                        output_path = os.path.join(output_dir, file)
                        plt.imsave(output_path, filtered_image, cmap='gray')
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print(f"Filtered and saved: {output_path} in {elapsed_time:.2f} seconds", flush=True)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}", flush=True)
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time
    print(f"Total processing time: {total_elapsed_time:.2f} seconds", flush=True)

def main():
    base_dir = "/data/bl70/validate/ProcessedImages/CBIS-DDSM/Segmented"
    output_base_dir = "/data/bl70/validate/FurtherProcessedImages/CBIS-DDSM"
    input_dirs = [
       # "SegmentedCL2.0_TG16x16_WN0.01",
        "SegmentedCL2.0_TG16x16_WN0.1",
       # "SegmentedCL2.0_TG16x16_WN1.0",
        "SegmentedCL2.0_TG8x8_WN0.01",
       # "SegmentedCL2.0_TG8x8_WN0.1",
        "SegmentedCL2.0_TG8x8_WN1.0",
     #   "SegmentedCL3.0_TG16x16_WN0.01",
       # "SegmentedCL3.0_TG16x16_WN0.1",
        "SegmentedCL3.0_TG16x16_WN1.0",
      #  "SegmentedCL3.0_TG8x8_WN0.01",
        #"SegmentedCL3.0_TG8x8_WN0.1",
        #"SegmentedCL3.0_TG8x8_WN1.0",
      #  "SegmentedCL5.0_TG16x16_WN0.01",
       # "SegmentedCL5.0_TG16x16_WN0.1",
        "SegmentedCL5.0_TG16x16_WN1.0",
        #"SegmentedCL5.0_TG8x8_WN0.01",
        #"SegmentedCL5.0_TG8x8_WN0.1",
        #"SegmentedCL5.0_TG8x8_WN1.0"
    ]


    input_dirs = [os.path.join(base_dir, d) for d in input_dirs]

    radius = 1  # Radius for the adaptive fuzzy median filter

    apply_additional_filtering(input_dirs, output_base_dir, radius)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Script terminated by user.", flush=True)
    except Exception as e:
        print(f"An error occurred: {e}", flush=True)
