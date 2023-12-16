"""The code in this file is used to reduce the size (press) iamges down to smaller sizes to address memory constraints for
    logistic regression."""

from PIL import Image
import os

def resize_images(input_folder, output_folder, target_size=(128, 128)):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get a list of all files in the input folder
    files = os.listdir(input_folder)

    for file in files:
        # Construct the full path of the input file
        input_path = os.path.join(input_folder, file)

        # Open the image using Pillow
        with Image.open(input_path) as img:
            # Resize the image to the target size
            resized_img = img.resize(target_size)

            # Construct the full path of the output file
            output_path = os.path.join(output_folder, file)

            # Save the resized image
            resized_img.save(output_path)

if __name__ == "__main__":
    # Specify input and output folders
    big_input_folder = "/Users/uochuba/Documents/Stanford/Senior/CS229/FOREST_DATA_WORK/trees_dataset/test"
    big_output_folder = "/Users/uochuba/Documents/Stanford/Senior/CS229/FOREST_DATA_WORK/test_128"

    # Call the resize_images function
    for input_folder in os.listdir(big_input_folder):
        joined_input_folder = os.path.join(big_input_folder, input_folder)
        output_folder = os.path.join(big_output_folder, input_folder)
        resize_images(joined_input_folder, output_folder)

    print("Image compression to 128x128 pixels complete.")