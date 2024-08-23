import os
import numpy as np
import tensorflow as tf

# Check if any GPUs are available
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        # Set memory growth to avoid allocating all GPU memory at once
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and will be used.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPU found. Running on CPU.")

def load_png_images_batch(path, start_idx, batch_size, processed_files):
    print(f"Loading batch from index {start_idx} with batch size {batch_size}")
    images = []
    file_names = sorted(os.listdir(path))
    for file in file_names[start_idx:start_idx + batch_size]:
        if file.endswith('.png') and file not in processed_files:
            filepath = os.path.join(path, file)
            print(f"Reading file: {filepath}")
            image = tf.io.read_file(filepath)
            image = tf.image.decode_png(image, channels=1)
            image = tf.image.convert_image_dtype(image, tf.float32)
            images.append((image, file))
    print(f"Loaded {len(images)} images")
    return images

def augment_image(image):
    print(f"Starting augmentation")
    augmented_images = []
    titles = []

    # Original image
    augmented_images.append(image)
    titles.append("Original")

    # Horizontal flipping
    flipped_horizontally = tf.image.flip_left_right(image)
    augmented_images.append(flipped_horizontally)
    titles.append("Original_Flipped_Horizontally")

    # Vertical flipping
    flipped_vertically = tf.image.flip_up_down(image)
    augmented_images.append(flipped_vertically)
    titles.append("Original_Flipped_Vertically")

    # Rotations
    angles = [45, 90, 135, 180, 225, 270]
    for angle in angles:
        rotated_image = tf.image.rot90(image, k=angle // 90)
        augmented_images.append(rotated_image)
        titles.append(f"Rotated_{angle}_degrees")

        # Apply flipping to rotated images only for 90 and 270 degrees
        if angle in [90, 270]:
            rotated_flipped_horizontally = tf.image.flip_left_right(rotated_image)
            augmented_images.append(rotated_flipped_horizontally)
            titles.append(f"Rotated_{angle}_degrees_Flipped_Horizontally")

            rotated_flipped_vertically = tf.image.flip_up_down(rotated_image)
            augmented_images.append(rotated_flipped_vertically)
            titles.append(f"Rotated_{angle}_degrees_Flipped_Vertically")

    # Scaling
    scales = [0.9, 1.1]
    for scale in scales:
        scaled_image = tf.image.resize(image, [int(image.shape[0] * scale), int(image.shape[1] * scale)])
        if scale > 1:
            scaled_image = tf.image.resize_with_crop_or_pad(scaled_image, image.shape[0], image.shape[1])
        else:
            scaled_image = tf.image.resize_with_crop_or_pad(scaled_image, image.shape[0], image.shape[1])
        augmented_images.append(scaled_image)
        titles.append(f"Scaled_{int(scale*100)}_percent")

    # Translation
    translations = [(-10, 10), (10, -10)]
    for t in translations:
        translated_image = tf.roll(image, shift=t, axis=[0, 1])
        augmented_images.append(translated_image)
        titles.append(f"Translated_{t[0]}_x_{t[1]}_y")

    # Stretching
    stretch_factors = [1.2, 0.8]
    for factor in stretch_factors:
        scaled_shape = [int(image.shape[0] * factor), image.shape[1]]
        stretched_image = tf.image.resize(image, scaled_shape)
        stretched_image = tf.image.resize_with_crop_or_pad(stretched_image, image.shape[0], image.shape[1])
        augmented_images.append(stretched_image)
        titles.append(f"Stretched_{int(factor*100)}_percent")

    # Shearing (approximation using affine transformations)
    shear_factors = [0.2, -0.2]
    for shear in shear_factors:
        shear_matrix = [[1, shear, 0], [0, 1, 0], [0, 0, 1]]
        sheared_image = tf.keras.preprocessing.image.apply_affine_transform(image.numpy(), shear=shear)
        augmented_images.append(tf.convert_to_tensor(sheared_image))
        titles.append(f"Sheared_{int(shear*100)}_percent")

    print(f"Finished augmentation, generated {len(augmented_images)} images")
    return augmented_images, titles

def save_images(images, titles, output_dir, original_file_name):
    print(f"Saving images to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(original_file_name)[0]  # Removes the extension from the original file name
    for image, title in zip(images, titles):
        filename = f"{base_name}_{title}.png".replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")
        filepath = os.path.join(output_dir, filename)
        print(f"Saving image: {filepath}")
        image_encoded = tf.image.encode_png(tf.image.convert_image_dtype(image, tf.uint8))
        tf.io.write_file(filepath, image_encoded)
        print(f"Saved: {filepath}")

if __name__ == "__main__":
  #  base_path = "/data/bl70/CBIS-DDSM/CBIS-DDSM/processed"
    base_path = "/data/bl70/CMMD/CMMD/processed"
  # output_base_dir = "/data/bl70/CBIS-DDSM/CBIS-DDSM/augmented"
    output_base_dir = "/data/bl70/CMMD/CMMD/augmented"
    batch_size = 1 # Adjust this based on your memory capacity

    print(f"Starting data augmentation script with base path: {base_path}")

    # Iterate through each folder in the base path
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path):
            print(f"Processing folder: {folder_name}")

            # Recursively iterate through subfolders to find images
            for root, _, files in os.walk(folder_path):
                output_dir = os.path.join(output_base_dir, os.path.relpath(root, base_path))
                os.makedirs(output_dir, exist_ok=True)

                processed_files = set()
                if os.path.exists(output_dir):
                    processed_files = set(os.path.splitext(f)[0] for f in os.listdir(output_dir) if f.endswith('.png'))

                file_names = sorted(f for f in files if f.endswith('.png'))
                num_files = len(file_names)
                print(f"Found {num_files} files in {root}")

                for start_idx in range(0, num_files, batch_size):
                    print(f"Processing batch starting at index {start_idx}")
                    batch = load_png_images_batch(root, start_idx, batch_size, processed_files)
                    for image, file_name in batch:
                       print(f"Augmenting image {file_name} in folder {root}:")
                       augmented_images, titles = augment_image(image)
    save_images(augmented_images, titles, output_dir, file_name)

    print("Data augmentation script completed.")
