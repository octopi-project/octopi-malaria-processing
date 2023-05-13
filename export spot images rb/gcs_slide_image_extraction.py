import subprocess
import os

bucket_name = “gs://octopi-malaria-uganda-2022-data”

folders_exported = ['Fingerprick_HL_2023-01-27_B01_2023-02-17_13-04-48.321619','Fingerprick_HL_2023-01-27_B02_2023-02-17_15-31-53.777898','Fingerprick_HL_2023-01-27_B03_2023-02-17_15-50-57.776833','Fingerprick_HL_2023-01-27_C02_2023-02-17_16-13-57.571627','Fingerprick_HL_2023-01-27_C03_2023-02-17_16-35-37.082487','SBC_2022-02-22_4uL-Med-A_2023-02-21_17-56-27.851194', 'SBC_2022-02-22_4uL-Med-B_2023-02-21_18-28-0.005527','SBC_2022-03-14_3uL-Med-C_2023-02-21_18-59-7.099675','SBC_2022-10-20_LIR01-C_2023-02-21_17-19-31.401096','SBC_2022-10-20_LIR01-D_2023-02-21_17-40-55.531778','PBC-1023-1_2023-01-22_19-59-54.633046','PAT-070-3_2023-01-22_15-24-28.812821','PBC-404-1_2023-01-22_19-09-9.267139','PAT-071-3_2023-01-22_15-47-3.096602','PBC-502-1_2023-01-22_17-49-38.429975','PAT-072-1_2023-01-22_17-17-58.363496','PBC-800-1_2023-01-22_21-30-44.794123','PAT-073-1_2023-01-22_16-32-5.192404','PBC-801-1_2023-01-22_22-06-18.047215','PAT-074-1_2023-01-22_16-55-50.887780']

# Run the gsutil ls command on the bucket and capture its output
folder_list = subprocess.check_output(['gsutil', 'ls', bucket_name])

# Convert the output to a list of strings without the prefix and trailing slash
sample_folders = [line.decode('utf-8').replace(bucket_name + '/', '').rstrip('/') for line in folder_list.splitlines()]

# Filter out the folders with the year equal to 2022
sample_folders = [folder for folder in sample_folders if folder.split(“_”)[-2].split(“-”)[0] != '2022']

# Get list of folders that haven't gone through spot processing yet
processing_bucket_name = f”{bucket_name}-processing”

# gsutil -q stat returns 0 if the folder IS present in the bucket, and 0 if not
folders_to_process_for_spots = [folder for folder in sample_folders if subprocess.call(f”gsutil -q stat {processing_bucket_name}/{folder}“, shell=True) != 0]

# save the folders to process to lists_of_datasets.txt in run_processing's folder
list_of_ds_file = '/home/hongquan/Desktop/Octopi/octopi-malaria/data processing/list of datasets.txt'
with open(list_of_ds_file, 'w') as outfile:
    for folder in folders_to_process_for_spots:
        outfile.write(folder + '\n')

# process the folders with run_processing.py
subprocess.run(['python3', '/home/hongquan/Desktop/Octopi/octopi-malaria/data processing/run_processing.py'])

# confirm every folder in sample_folders is processed
folders_to_process_confirm = [folder for folder in sample_folders if subprocess.call(f”gsutil -q stat {processing_bucket_name}/{folder}“, shell=True) != 0]
print('There are ' + str(len(folders_to_process_confirm)) + ' folders left that have not been processed for spot extraction')

# next, remove the folders for which we already have the .npy files
npy_files_dir = '/media/hongquan/Extreme SSD/' # where I expect the .npy files to be saved
sample_folders = [folder for folder in sample_folders if not os.path.exists(os.path.join(ssd_dir, folder + '.npy'))]

# then, add all remaining folders in sample_folders to lists_of_datasets.txt in export_spot_images's folder
list_of_ds_file = '/home/hongquan/Desktop/Octopi/octopi-malaria/export spot images for classification/list of datasets.txt'
with open(list_of_ds_file, 'w') as outfile:
    for folder in sample_folders:

        outfile.write(folder + '\n')

# now export all spot images from sample_folders
subprocess.run([“python3”, “/home/hongquan/Desktop/Octopi/octopi-malaria/export spot images for classification/export_spot_images.py”])

# move the spot images .npy files over to SSD
for f in sample_folders:
    src_path = f'{f}.npy'
    dst_path = f'{npy_files_dir}/{src_path}'
    subprocess.run(['mv', src_path, dst_path])


