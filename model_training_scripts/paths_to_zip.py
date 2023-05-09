# save the files to zip file to process on my computer
import zipfile
import os 

data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/'
folders = ['s_a','s_b','s_1a','s_1b','s_2a','s_2b','s_3a','s_3b']

paths = []
for folder in folders:
    path = data_dir + folder
    if len(folder) == 3:
        path += '/ann_with_predictions_r18_b32.csv'
        paths.append(path)
    else:
        path1 = path + '/ann_with_predictions_cl_r18_b32.csv'
        path2 = path + '/unsure_r18_b32_relabeled_thresh_0.9.csv'
        paths.append(path1)
        paths.append(path2)

# Create the zip file and add the csv files to it
with zipfile.ZipFile(data_dir + "/ann_with_predictions_r18_b32.zip", "w") as zip_file:
    for path in paths:
        # Get the name of the nested directory that the csv file is in
        nested_directory = os.path.split(os.path.dirname(path))[1]
        # Construct the path to where the csv file should be inside the zip file
        zip_path = os.path.join(nested_directory, os.path.basename(path))
        # Add the csv file to the zip file
        zip_file.write(path, arcname=zip_path)