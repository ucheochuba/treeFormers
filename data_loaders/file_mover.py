"""The code in this file is used to extract ALL images from the Stanford ML Group, and place it into a folder that 
    can be easily loaded into the transformer"""

import pandas as pd
from PIL import Image
import os
import shutil

csv_path = 'ForestNetDataset/val.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_path)

# Specify input and output directories
input_base_dir = 'ForestNetDataset'  # Replace with the path to your input images directory
output_base_dir = 'val'  # Replace with the path to your output images directory

# Create output directories based on labels
labels = df['merged_label'].unique()
for label in labels:
    output_label_dir = os.path.join(output_base_dir, str(label))
    os.makedirs(output_label_dir, exist_ok=True)

# Process each row in the CSV
for index, row in df.iterrows():
    # Extract information from the CSV
    folder_path = row['example_path']
    label = row['merged_label']

    inner_folder_path = os.path.join(input_base_dir, folder_path, 'images', 'visible')

    # Iterate through all PNG files in the inner folder
    for file_name in os.listdir(inner_folder_path):
        if file_name.lower().endswith('.png'):
            # Construct input and output paths
            # breakpoint()
            input_file_path = os.path.join(inner_folder_path, file_name)
            output_file_name = f"{index}_{file_name.replace('.png', '')}_image.jpg"
            output_label_dir = os.path.join(output_base_dir, str(label))
            output_file_path = os.path.join(output_label_dir, output_file_name)

            # Open and convert the image
            with Image.open(input_file_path) as img:
                img = img.convert('RGB')
                img.save(output_file_path)

            # Optionally, you can remove the original PNG file if needed
            # os.remove(input_file_path)

print("Conversion and moving completed.")