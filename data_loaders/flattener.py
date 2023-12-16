"""The code in this file was used to flatten the images into an RGB pixel intensity format, on which we perform logistic regression."""

import os
import pandas as pd
from PIL import Image
import numpy as np
import csv

image_size = 3 * 128 * 128

# Function to get image data
def get_image_data(image_path, label):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
        rgb_values = np.array(img).flatten()
        image_data = np.insert(rgb_values, 0, label)
        return image_data

# Main function to process the images and save to CSV
def process_images_to_csv(base_dir):
    # List to hold all image data
    all_image_data = []
    
    # Labels (folder names) and corresponding integer class
    labels = sorted(os.listdir(base_dir))
    label_to_int = {label: index for index, label in enumerate(labels)}
    
    # Process each folder (label)
    for label in labels:
        class_dir = os.path.join(base_dir, label)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image_label = label_to_int[label]
            image_row = get_image_data(image_path, image_label)
            all_image_data.append(image_row)
    
    # Convert to a DataFrame
    num_pixels = image_size
    column_names = ['label'] + [f'pixel{i}' for i in range(num_pixels)]
    with open('tree_sheet.csv', 'w', newline='') as csv_file:
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)

        # Write the header row
        csv_writer.writerow(column_names)

        # Write the remaining rows
        for row in all_image_data:
            csv_writer.writerow(row)

    print(f"CSV file created successfully.")

base_dir = '/Users/uochuba/Documents/Stanford/Senior/CS229/FOREST_DATA_WORK/trees_dataset_128/test_128'
process_images_to_csv(base_dir)