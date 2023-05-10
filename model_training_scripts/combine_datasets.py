import numpy as np
import pandas as pd
import os

# GLOBAL VARIABLES
combine_dict = {} # dictionary with image path: ann path
combine_dict['/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_parasite.npy'] = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_parasite.csv'
combine_dict['/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_unsure.npy'] = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_unsure.csv'
combine_dict['/home/rinni/Desktop/Octopi/data/neg-to-combine/neg_combined_images.npy'] = '/home/rinni/Desktop/Octopi/data/neg-to-combine/neg_combined_ann.csv'

out_im_path = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_for_training_performance.npy' # all images except patient negatives
out_ann_path = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_for_training_performance.csv'


# combines the datasets in combine_dict; saves the outputs to the given image and annotation files
def combine_datasets(combine_dict, out_image_path, out_ann_path):
	images = []
	annotations = []
	# add the images / annotations for each dictionary element to their respective lists
	for npy_path in list(combine_dict.keys()):
		images_ = np.load(npy_path)
		ann_pd = pd.read_csv(combine_dict[npy_path],index_col='index')
		ann_pd = ann_pd.sort_index()
		annotations_ = ann_pd['annotation'].values.squeeze()

		images.append(images_)
		annotations.append(annotations_)
	
	# combine
	images = np.concatenate(images, axis=0)
	print('The combined images have shape: '); print(images.shape)

	annotations = np.concatenate(annotations)
	comb_ann_df = pd.DataFrame({'annotation':annotations})
	comb_ann_df.index.name = 'index'

	# save
	np.save(out_image_path, images)
	comb_ann_df.to_csv(out_ann_path)


combine_datasets(combine_dict, out_im_path, out_ann_path)