import shutil
import os

# get images to combine from lists of datasets that were just processed
f = open('/home/rinni/octopi-malaria/export spot images rb/pos list of datasets.txt','r')
# f = open('/home/rinni/octopi-malaria/export spot images rb/neg list of datasets.txt','r')
DATASET_ID = f.read()
DATASET_ID = DATASET_ID.split('\n')
DATASET_ID = DATASET_ID[:-1] # remove empty string at end
f.close()
dir_in = '/media/rinni/Extreme SSD/Rinni/Octopi/data/'
# dir_out = dir_in + 'neg_testing_slides/'
dir_out = dir_in + 'pos_testing_slides/'
if not os.path.exists(dir_out):
    os.makedirs(dir_out)

for dataset in DATASET_ID:
    # Get the filename from the full path
    filename = dataset + '.npy'
    
    # Create the source and destination paths
    src = os.path.join(dir_in, filename)
    dst = os.path.join(dir_out, filename)
    
    # Move the file from the source to the destination
    shutil.move(src, dst)