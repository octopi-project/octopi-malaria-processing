import numpy as np
import pandas as pd
import os

# get ids for all images labeled unsure
def isolate_unsure(annotation_pd, images):
	idx = annotation_pd['annotation'].isin([2])
	idx_not = annotation_pd['annotation'].isin([0,1])

	annotation_pd_p = annotation_pd.loc[idx]
	annotations_ = annotation_pd_p['annotation'].values.squeeze()
	images_p = images[idx,]
	annotation_not_pd = annotation_pd.loc[idx_not]
	annotations_not_ = annotation_not_pd['annotation'].values.squeeze()
	images_not = images[idx_not,]
	return annotations_, images_p, annotations_not_, images_not

# get ids for all negative images that were predicted to be unsure or parasites
def isolate_mislabeled_negatives(annotation_pd, images):
	cond_annotation = annotation_pd['annotation'] == 0
	cond_labeled_unsure = annotation_pd['unsure output'] > 0.5
	cond_labeled_par = annotation_pd['parasite output'] > 0.5
	cond = cond_annotation & (cond_labeled_unsure | cond_labeled_par)
	cond_not = cond_annotation & ~(cond_labeled_unsure | cond_labeled_par)
	idx = annotation_pd.loc[cond].index
	idx_not = annotation_pd.loc[cond_not].index

	annotation_pd_p = annotation_pd.loc[idx]
	annotations_ = annotation_pd_p['annotation'].values.squeeze()
	images_p = images[idx,]
	annotation_not_pd = annotation_pd.loc[idx_not]
	annotations_not_ = annotation_not_pd['annotation'].values.squeeze()
	images_not = images[idx_not,]
	return annotations_, images_p, annotations_not_, images_not

# get ids for all positive images that were predicted to be unsure or negatives
def isolate_mislabeled_positives(annotation_pd, images):
	cond_annotation = annotation_pd['annotation'] == 1
	cond_labeled_unsure = annotation_pd['unsure output'] > 0.5
	cond_labeled_nonpar = annotation_pd['non-parasite output'] > 0.5
	cond = cond_annotation & (cond_labeled_unsure | cond_labeled_nonpar)
	cond_not = cond_annotation & ~(cond_labeled_unsure | cond_labeled_nonpar)
	idx = annotation_pd.loc[cond].index
	idx_not = annotation_pd.loc[cond_not].index

	annotation_pd_p = annotation_pd.loc[idx]
	annotations_ = annotation_pd_p['annotation'].values.squeeze()
	images_p = images[idx,]
	annotation_not_pd = annotation_pd.loc[idx_not]
	annotations_not_ = annotation_not_pd['annotation'].values.squeeze()
	images_not = images[idx_not,]
	return annotations_, images_p, annotations_not_, images_not

# folder containing all image and annotations files to filter
data_dir = '/home/rinni/Desktop/Octopi/data/to-combine/'

dataset_npy_ids = [] # names of all .npy files
for file in os.listdir(data_dir):
	if file.endswith('.npy'):
		if 'combine' not in file:
			dataset_npy_ids.append(data_dir + file)

images = []
annotations = []
images_not = []
annotations_not = []

for npy_id_ in dataset_npy_ids:

	images_ = np.load(npy_id_)

	annotation_id_ = data_dir + 'combined_annotations_with_m502_predictions_original.csv' # assumes the annotations file has the same name with _annotations_with_predictions at the end
	if os.path.exists(annotation_id_):
		annotation_pd = pd.read_csv(annotation_id_,index_col='index')
		annotation_pd = annotation_pd.sort_index()
		annotations_, images_, annotations_not_, images_not_ = isolate_unsure(annotation_pd, images_)
	else:
		print('Can\'t find annotation file: ' + annotation_id_)

	print(images_.shape)
	images.append(images_)
	annotations.append(annotations_)
	images_not.append(images_not_)
	annotations_not.append(annotations_not_)

images = np.concatenate(images,axis=0)
annotations = np.concatenate(annotations)
images_not = np.concatenate(images_not,axis=0)
annotations_not = np.concatenate(annotations_not)

print(images.shape)

np.save(data_dir + '/combined_images_unsure.npy',images)
np.save(data_dir + '/combined_images_pos.npy',images_not)
df = pd.DataFrame({'annotation':annotations})
df.index.name = 'index'
df.to_csv(data_dir + '/combined_annotations_unsure.csv')
df_not = pd.DataFrame({'annotation':annotations_not})
df_not.index.name = 'index'
df_not.to_csv(data_dir + '/combined_annotations_pos.csv')