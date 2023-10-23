import subprocess
import os

bucket_name = "gs://octopi-malaria-uganda-2022-data"

# # Run the gsutil ls command on the bucket and capture its output
# folder_list = subprocess.check_output(['gsutil', 'ls', bucket_name])
# # folder_list = folder_list.decode().strip().split("\n")
# # print(folder_list[0]);print(folder_list[1])

# # Convert the output to a list of strings without the prefix and trailing slash
# sample_folders = [line.decode('utf-8').replace(bucket_name + '/', '').rstrip('/') for line in folder_list.splitlines()]

# # Filter out the folders with the year equal to 2022
# sample_folders = [folder for folder in sample_folders if folder.split('_')[-2].split('-')[0] != '2022']
# print('First set of folders to process:')
# print('\n'.join(sample_folders))

# # Get list of folders that haven't gone through spot processing yet
# processing_bucket_name = f"{bucket_name}-processing"

# # gsutil -q stat returns 0 if the folder IS present in the bucket, and 0 if not
# '''
# folders_to_process_for_spots = [folder for folder in sample_folders if subprocess.call(f"gsutil -q stat {processing_bucket_name}/{folder}/*", shell=True) != 0]
# folders_to_process_for_spots = [folder for folder in folders_to_process_for_spots if folder not in folders_exported]
# print('\a')
# print('Folders to process for spots:')
# print('\n'.join(folders_to_process_for_spots))

# # run processing if needed
# if len(folders_to_process_for_spots) != 0:
#     # change dir
#     subprocess.run(["cd", "home/rinni/octopi-malaria/data processing/"], shell=True)
#     # save the folders to process to lists_of_datasets.txt in run_processing's folder
#     list_of_ds_file = '/home/rinni/octopi-malaria/data processing/list of datasets.txt'
#     with open(list_of_ds_file, 'w') as outfile:
#         for folder in folders_to_process_for_spots:
#             outfile.write(folder + '\n')

#     # process the folders with run_processing.py
#     subprocess.run(['python3', '/home/rinni/octopi-malaria/data processing/run_processing.py'])

# # confirm every folder in sample_folders is processed
# folders_to_process_confirm = [folder for folder in sample_folders if subprocess.call(f"gsutil -q stat {processing_bucket_name}/{folder}/*", shell=True) != 0]
# folders_to_process_confirm = [folder for folder in folders_to_process_confirm if folder not in folders_exported]
# print('There are ' + str(len(folders_to_process_confirm)) + ' folders left that have not been processed for spot extraction')
# '''

# # then, add all folders in sample_folders to lists_of_datasets.txt in export_spot_images's folder
# # change dir
# if len(sample_folders) != 0:
#     directory_path = "/home/rinni/octopi-malaria/export spot images rb/"
#     # Change the current working directory to the specified directory path
#     subprocess.run(f"cd '{directory_path}' && pwd", shell=True, executable='/bin/bash')

#     list_of_ds_file = '/home/octopi/Desktop/Octopi/Notebook/all-test/list-of-datasets-to-export.txt'
#     with open(list_of_ds_file, 'w') as outfile:

# now export all spot images from sample_folders
subprocess.run(['python3', '/home/octopi/Desktop/octopi-malaria/export spot images rb/export_spot_images.py'])

# # move the spot images .npy files over to SSD
# for f in sample_folders:
#     src_path = f'{f}.npy'
#     dst_path = f'media/rinni/Extreme SSD/Rinni/Octopi/data/{src_path}'
#     subprocess.run(['mv', src_path, dst_path])


