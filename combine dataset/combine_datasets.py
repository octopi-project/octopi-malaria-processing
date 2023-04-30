import numpy as np
import pandas as pd
import os

# folder containing all image and annotations files to combine
data_dir = '/home/rinni/Desktop/Octopi/data/to-combine/'

# dataset_npy_ids = [] # names of all .npy files
# for file in os.listdir(data_dir):
# 	if file.endswith('.npy'):
# 		if file != 'combined_images.npy':
# 			dataset_npy_ids.append(data_dir + file)

dataset_npy_ids = {'combined_images_unsure.npy':'combined_annotations_unsure_only_labeled_pos.csv', 'combined_images_mislabeled_neg.npy':'combined_annotations_mislabeled_neg.csv'}
images = []
annotations = []

for npy_id_ in list(dataset_npy_ids.keys()):

	images_ = np.load(data_dir + npy_id_)

	annotation_id_ = data_dir + dataset_npy_ids[npy_id_]
	# annotation_id_ = os.path.splitext(npy_id_)[0] + '_annotations.csv' # assumes the annotations file has the same name with _annotations at the end
	if os.path.exists(annotation_id_):
		annotation_pd = pd.read_csv(annotation_id_,index_col='index')
		annotation_pd = annotation_pd.sort_index()
		# idx = annotation_pd['annotation'].isin([0, 1, 2]) # save all annotated images 
		idx = annotation_pd['annotation'].isin([0, 1]) # save positive and negative images only
		images_ = images_[idx,]
		annotation_pd = annotation_pd[idx]
		annotations_ = annotation_pd['annotation'].values.squeeze()
	else:
		print('Can\'t find annotation file: ' + annotation_id_)

	print(images_.shape)
	images.append(images_)
	annotations.append(annotations_)

images = np.concatenate(images,axis=0)
annotations = np.concatenate(annotations)

print(images.shape)

np.save(data_dir + '/combined_images_unsure_as_pos_and_mislabeled_neg.npy',images)
df = pd.DataFrame({'annotation':annotations})
df.index.name = 'index'
df.to_csv(data_dir + '/combined_ann_unsure_as_pos_and_mislabeled_neg.csv')