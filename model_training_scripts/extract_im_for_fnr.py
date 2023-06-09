import os
import random
import numpy as np

# get slides
# Specify the path to the directory containing the folders
path = "/path/to/directory"
# Get a list of all files in the specified path
files = [file for file in os.listdir(path) if file.endswith(".npy")]
# Filter out files containing the substring "SBC"
filtered_files = [file for file in files if "SBC" not in file]
# Randomly select 3 files from the filtered list
random_files = random.sample(filtered_files, 3)

# get random images from each slide
for slide in random_files:
    # Load the npy file
    data = np.load(slide + ".npy")
    # Get the total number of images
    num_images = data.shape[0]
    # Randomly select 1000 indices without replacement
    random_indices = np.random.choice(num_images, size=1000, replace=False)
    # Select the corresponding images
    random_images = data[random_indices]
    # Save the selected images to a new npy file
    np.save(slide + "_selected_images.npy", random_images)