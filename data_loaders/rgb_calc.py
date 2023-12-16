"""The code in this file was used to calculate the mean and std of pixel color intensities across the dataset to apply normalization
    to the custom built (scratch) transformer."""

from PIL import Image
import os
import numpy as np

def calculate_mean_std(folder_path):
    # Initialize accumulators
    mean_r, mean_g, mean_b = 0, 0, 0
    std_r, std_g, std_b = 0, 0, 0
    total_images = 0

    # Iterate through each image in the folder
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            image_path = os.path.join(root, file)
            img = Image.open(image_path)
            img_array = np.array(img) / 255.0  # Normalize to 0-1 range

            # Calculate mean and std for each channel
            mean_r += np.mean(img_array[:, :, 0])
            mean_g += np.mean(img_array[:, :, 1])
            mean_b += np.mean(img_array[:, :, 2])

            std_r += np.std(img_array[:, :, 0])
            std_g += np.std(img_array[:, :, 1])
            std_b += np.std(img_array[:, :, 2])

            total_images += 1

    # Calculate overall mean and std
    mean_r /= total_images
    mean_g /= total_images
    mean_b /= total_images

    std_r /= total_images
    std_g /= total_images
    std_b /= total_images

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)

# Specify the paths to train, val, and test folders
train_path = 'train'
val_path = 'val'
test_path = 'test'

# Calculate mean and std for each set
train_mean, train_std = calculate_mean_std(train_path)
val_mean, val_std = calculate_mean_std(val_path)
test_mean, test_std = calculate_mean_std(test_path)

# Print the results
print(f'Train Mean: {train_mean}, Train Std: {train_std}')
print(f'Validation Mean: {val_mean}, Validation Std: {val_std}')
print(f'Test Mean: {test_mean}, Test Std: {test_std}')