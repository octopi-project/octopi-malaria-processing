import numpy as np
import os

# get images to combine from lists of datasets that were just processed
# f = open('/home/rinni/octopi-malaria/export spot images rb/pos list of datasets.txt','r')
f = open('/home/rinni/octopi-malaria/export spot images rb/neg list of datasets.txt','r')
DATASET_ID = f.read()
DATASET_ID = DATASET_ID.split('\n')
DATASET_ID = DATASET_ID[:-1] # remove empty string at end
f.close()

top_dir = '/media/rinni/Extreme SSD/Rinni/Octopi/data/'
counter = 0
for dataset in DATASET_ID:
	print('Adding ' + dataset)
	if counter == 0:
		im_big = np.load(top_dir + dataset + '.npy')
	else:
		im = np.load(top_dir + dataset + '.npy')
		im_big = np.concatenate((im_big, im), axis=0)
	counter += 1

# np.save(top_dir + '/combined_pos_test.npy',im_big)
np.save(top_dir + '/combined_neg_test.npy',im_big)