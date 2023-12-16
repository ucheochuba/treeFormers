"""The code in this file is used to extract only the composite images from the Stanford ML Group, and place it into a folder that 
    can be easily loaded into the transformer"""

import pandas as pd
from PIL import Image
import os
import shutil

mode = 'val' # 'train', 'val', or 'test'

csv_path = f'ForestNetDataset/{mode}.csv'  # Replace with the path to your CSV file
df = pd.read_csv(csv_path)

# Specify input and output directories
input_base_dir = 'ForestNetDataset'
output_base_dir = mode

# Create output directories based on labels
labels = df['merged_label'].unique()
for label in labels:
    output_label_dir = os.path.join(output_base_dir, str(label))
    os.makedirs(output_label_dir, exist_ok=True)

# Process each row in the CSV
for index, row in df.iterrows():
    # Extract information from the CSV
    file_path = row['example_path']
    label = row['merged_label']

    file_path += '/images/visible/composite.png'

    # Construct input and output paths
    input_file_path = os.path.join(input_base_dir, file_path)
    output_file_name = f"{index}_image.jpg"
    output_label_dir = os.path.join(output_base_dir, str(label))
    output_file_path = os.path.join(output_label_dir, output_file_name)

    # Open and convert the image
    with Image.open(input_file_path) as img:
        img = img.convert('RGB')
        img.save(output_file_path)

print("Conversion and moving completed.")