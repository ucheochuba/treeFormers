"""The code in this file modifies images to add horizontal bars to the images with intensities representing longitudinal data,
    as discussed in the TreeFormers paper."""

import pandas as pd
from PIL import Image
import os
import shutil

from PIL import Image, ImageDraw

# Draws bar representing longitude
def add_hor_bar(original_image, bar_intensity, bar_height):
    # Create a new image with the same width and height as the original plus the bar
    new_width, new_height = original_image.size[0], original_image.size[1] + bar_height
    new_image = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

    # Paste the original image onto the new image
    new_image.paste(original_image, (0, 0))

    # Create a grayscale bar
    bar = Image.new("L", (new_width, bar_height), color=bar_intensity)

    # Paste the grayscale bar onto the new image at the bottom
    new_image.paste(bar, (0, original_image.size[1]))

    return new_image

# Draws bar representing latitude
def add_ver_bar(original_image, bar_intensity, bar_width):
    # Create a new image with the same width and height as the original plus the bar
    new_width, new_height = original_image.size[0] + bar_width, original_image.size[1]
    new_image = Image.new("RGB", (new_width, new_height), color=(255, 255, 255))

    # Paste the original image onto the new image
    new_image.paste(original_image, (0, 0))

    # Create a grayscale bar
    bar = Image.new("L", (bar_width, new_height), color=bar_intensity)

    # Paste the grayscale bar onto the new image at the bottom
    new_image.paste(bar, (original_image.size[0], 0))

    return new_image


csv_path = 'ForestNetDataset/test.csv'
df = pd.read_csv(csv_path)

# Specify input and output directories
input_base_dir = 'ForestNetDataset'
output_base_dir = 'test'

# Create output directories based on labels
labels = df['merged_label'].unique()
for label in labels:
    output_label_dir = os.path.join(output_base_dir, str(label))
    os.makedirs(output_label_dir, exist_ok=True)

# constants to normalize data
min_lat, lat_range, min_long, long_range = df['latitude'].min(), df['latitude'].max() - df['latitude'].min(), df['longitude'].min(), df['longitude'].max() - df['longitude'].min()


# Process each row in the CSV
for index, row in df.iterrows():
    # Extract information from the CSV
    file_path = row['example_path']
    label = row['merged_label']

    file_path += '/images/visible'

    # Iterate through all PNG files in the inner folder
    for file_name in os.listdir('ForestNetDataset/' + file_path):
        if file_name.lower().endswith('.png'):
            # Construct input and output paths
            input_file_path = os.path.join(input_base_dir, file_path)
            output_file_name = f"{index}_image.jpg"
            output_label_dir = os.path.join(output_base_dir, str(label))
            output_file_path = os.path.join(output_label_dir, f'{file_name}_{index}.png')
            lat_color = (row['latitude'] - min_lat) / lat_range
            long_color = (row['longitude'] - min_long) / long_range

            # Open and convert the image
            
            with Image.open('ForestNetDataset/' + file_path + '/' + file_name) as img:
                img = img.convert('RGB')
                img = add_ver_bar(img, int(long_color*255), 64)
                img = add_hor_bar(img, int(lat_color*255), 64)
                img.save(output_file_path)

print("Conversion and moving completed.")