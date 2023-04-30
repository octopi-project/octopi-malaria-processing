import numpy as np
import pandas as pd
import os

# get ids for all images labeled uns
def isolate_classes(annotation_pd, images):
	idx_n = annotation_pd['annotation'].isin([0])
	idx_p = annotation_pd['annotation'].isin([1])
	idx_u = annotation_pd['annotation'].isin([2])

	annotation_pd_n = annotation_pd.loc[idx_n]
	annotations_n = annotation_pd_n['annotation'].values.squeeze()
	images_n = images[idx_n,]

	annotation_pd_p = annotation_pd.loc[idx_p]
	annotations_p = annotation_pd_p['annotation'].values.squeeze()
	images_p = images[idx_p,]

	annotation_pd_u = annotation_pd.loc[idx_u]
	annotations_u = annotation_pd_u['annotation'].values.squeeze()
	images_u = images[idx_u,]
	return annotations_n, images_n, annotations_p, images_p, annotations_u, images_u

# get ids for all neg images that were predicted to be uns or parasites
def isolate_misclassified_neg(annotation_pd, images):
	cond_annotation = annotation_pd['annotation'] == 0
	cond_labeled_notneg = annotation_pd['non-parasite output'] < 0.5
	cond = cond_annotation & cond_labeled_notneg
	cond_not = cond_annotation & ~cond_labeled_notneg
	idx = annotation_pd.loc[cond].index
	idx_not = annotation_pd.loc[cond_not].index

	annotation_pd_p = annotation_pd.loc[idx]
	annotations_ = annotation_pd_p['annotation'].values.squeeze()
	images_p = images[idx,]
	annotation_not_pd = annotation_pd.loc[idx_not]
	annotations_not_ = annotation_not_pd['annotation'].values.squeeze()
	images_not = images[idx_not,]
	return annotations_, images_p, annotations_not_, images_not

# get ids for all pos images that were predicted to be uns or neg
def isolate_misclassified_pos(annotation_pd, images):
	cond_annotation = annotation_pd['annotation'] == 1
	cond_labeled_notpos = annotation_pd['parasite output'] < 0.5
	cond = cond_annotation & cond_labeled_notpos
	cond_not = cond_annotation & ~cond_labeled_notpos
	idx = annotation_pd.loc[cond].index
	idx_not = annotation_pd.loc[cond_not].index

	annotation_pd_p = annotation_pd.loc[idx]
	annotations_ = annotation_pd_p['annotation'].values.squeeze()
	images_p = images[idx,]
	annotation_not_pd = annotation_pd.loc[idx_not]
	annotations_not_ = annotation_not_pd['annotation'].values.squeeze()
	images_not = images[idx_not,]
	return annotations_, images_p, annotations_not_, images_not

def relabel_uns_likely_pos(diff_annotation_w_pred_path, all_unsure_idx, diff_relabled_path):
	diff_annotation_pd = pd.read_csv(diff_annotation_w_pred_path + '.csv', index_col='index')
	cond_annotation = diff_annotation_pd.index.isin(all_unsure_idx)
	cond_pos = diff_annotation_pd['parasite output'] > 0.9
	cond = cond_annotation & cond_pos
	cond_not = cond_annotation & ~cond_pos

	diff_annotation_pd.loc[cond, 'annotation'] = 1
	diff_annotation_pd.loc[cond_not, 'annotation'] = 0
	diff_annotation_pd = diff_annotation_pd.loc[:, ~diff_annotation_pd.columns.str.contains('output')]
	diff_annotation_pd.to_csv(diff_relabled_path + '.csv')

def combine_datasets(combine_dict, image_save_path, ann_save_path):
	images = []
	ann = []
	idxs = []
	for npy_id in list(combine_dict.keys()):
		images_ = np.load(npy_id + '.npy')
		annotation_pd = pd.read_csv(combine_dict[npy_id] + '.csv',index_col='index')
		annotations_ = annotation_pd['annotation'].values.squeeze()

		print(images_.shape)
		images.append(images_)
		ann.append(annotations_)
		idxs.append(images_.shape[0])
	images = np.concatenate(images, axis=0)
	ann = np.concatenate(ann)
	print(images.shape)

	np.save(image_save_path, images)
	df = pd.DataFrame({'annotation':ann})
	df.index.name = 'index'
	df.to_csv(ann_save_path + '.csv')

	return idxs









# FILTERING

# set up images and annotations to filter

# folder containing all image and annotations files to filter
data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/'
data_dir_base = data_dir.split('sorted_images')[0]
# folder containing all neg image and annotations files to filter
data_dir_n = '/home/rinni/Desktop/Octopi/data/neg-to-combine/'

# get images and annotations
npy_id = data_dir + 'combined_images.npy'
images_all = np.load(npy_id_)

annotation_id = data_dir + 'combined_annotations_with_predictions.csv'
annotation_pd = pd.read_csv(annotation_id,index_col='index')
annotation_pd = annotation_pd.sort_index()

npy_id_n = data_dir_n + 'neg_combined_images.npy'
images_all_n = np.load(npy_id_n)

annotation_id_n = data_dir_n + 'neg_combined_annotations_with_predictions.csv'
annotation_pd_n = pd.read_csv(annotation_id_n,index_col='index')
annotation_pd_n = annotation_pd_n.sort_index()

# split patient slides data into uns, neg, and pos datasets

annotations_n, images_n, annotations_p, images_p, annotations_u, images_u = isolate_classes(annotation_pd, images_all)

print(images_n.shape)

np.save(data_dir + '/combined_images_neg.npy',images_n)
df_n = pd.DataFrame({'annotation':annotations_n})
df_n.index.name = 'index'
df_n.to_csv(data_dir + '/combined_annotations_neg.csv')

np.save(data_dir + '/combined_images_pos.npy',images_p)
df_p = pd.DataFrame({'annotation':annotations_p})
df_p.index.name = 'index'
df_p.to_csv(data_dir + '/combined_annotations_pos.csv')

np.save(data_dir + '/combined_images_uns.npy',images_u)
df_u = pd.DataFrame({'annotation':annotations_u})
df_u.index.name = 'index'
df_u.to_csv(data_dir + '/combined_annotations_uns.csv')

# save uns labeled as pos & uns labeled as neg datasets

df_u['annotation'] = 1
df_u.to_csv(data_dir + '/combined_annotations_uns_labeled_pos.csv')

df_u['annotation'] = 0
df_u.to_csv(data_dir + '/combined_annotations_uns_labeled_neg.csv')

df_u['annotation'] = -1
df_u.to_csv(data_dir + '/combined_annotations_uns_unlabeled.csv')

# save misclassified neg and correct neg datasets
annotations_n_wrong, images_n_wrong, annotations_n_right, images_n_right = isolate_misclassified_neg(annotation_pd, images_all)

print(images_n_wrong.shape)

np.save(data_dir + '/combined_images_neg_misclassified.npy',images_n_wrong)
df_n_wrong = pd.DataFrame({'annotation':annotations_n_wrong})
df_n_wrong.index.name = 'index'
df_n_wrong.to_csv(data_dir + '/combined_annotations_neg_misclassified.csv')

np.save(data_dir + '/combined_images_neg_correct.npy',images_n_right)
df_n_right = pd.DataFrame({'annotation':annotations_n_right})
df_n_right.index.name = 'index'
df_n_right.to_csv(data_dir + '/combined_annotations_neg_correct.csv')

# save misclassified pos and correct pos datasets
annotations_p_wrong, images_p_wrong, annotations_p_right, images_p_right = isolate_misclassified_pos(annotation_pd, images_all)

print(images_p_wrong.shape)

np.save(data_dir + '/combined_images_pos_misclassified.npy',images_p_wrong)
df_p_wrong = pd.DataFrame({'annotation':annotations_p_wrong})
df_p_wrong.index.name = 'index'
df_p_wrong.to_csv(data_dir + '/combined_annotations_pos_misclassified.csv')

np.save(data_dir + '/combined_images_pos_correct.npy',images_n_right)
df_p_right = pd.DataFrame({'annotation':annotations_p_right})
df_p_right.index.name = 'index'
df_p_right.to_csv(data_dir + '/combined_annotations_pos_correct.csv')

# save misclassified neg and correct neg from neg slides

annotations_n_wrong, images_n_wrong, annotations_n_right, images_n_right = isolate_misclassified_neg(annotation_pd_n, images_all_n)

print(images_n_wrong.shape)

np.save(data_dir_n + '/neg_combined_images_misclassified.npy',images_n_wrong)
df_n_wrong = pd.DataFrame({'annotation':annotations_n_wrong})
df_n_wrong.index.name = 'index'
df_n_wrong.to_csv(data_dir_n + '/neg_combined_annotations_misclassified.csv')

np.save(data_dir_n + '/neg_combined_images_correct.npy',images_n_right)
df_n_right = pd.DataFrame({'annotation':annotations_n_right})
df_n_right.index.name = 'index'
df_n_right.to_csv(data_dir_n + '/neg_combined_annotations_correct.csv')

# COMBINING: for differentiators

# differentiator 1: uns as pos, the neg misclassified from neg slides

combine_dict = {data_dir + '/combined_images_uns': data_dir + '/combined_annotations_uns_labeled_pos'}
combine_dict[data_dir_n + '/neg_combined_images_misclassified'] = data_dir_n + 'neg_combined_annotations_misclassified'

idx_diff1 = combine_datasets(combine_dict, data_dir + '/diff1/combined_images_diff1', data_dir + '/diff1/combined_ann_diff1')
num_uns = idx_diff1[0]

# differentiator 2: uns as pos, the pos misclassified from pos slides, the neg misclassified from neg slides

# add to old dict
combine_dict[data_dir + '/combined_images_pos_misclassified'] = data_dir + 'combined_annotations_pos_misclassified'

idx_diff2 = combine_datasets(combine_dict, data_dir + '/diff2/combined_images_diff2', data_dir + '/diff2/combined_ann_diff2')

# differentiator 3: uns as uns, the pos misclassified from pos slides, the neg misclassified from neg slides

# modify old dict again
combine_dict[data_dir + '/combined_images_uns'] = data_dir + 'combined_annotations_uns_unlabeled'
print(combine_dict)

idx_diff3 = combine_datasets(combine_dict, data_dir + '/diff3/combined_images_diff3', data_dir + '/diff3/combined_ann_diff3')

# COMBINING: for classifiers

# classifier 0: combined original (no change in annotations) -- already done

# classifier 1: 
# using diff 1: uns labeled as pos with probability >0.9 and neg otherwise
# and all pos/neg from pos slides; all neg from neg slides

# to add: diff1 combined (and relabeled); pos/neg only; correct neg

idx_uns = np.arange(0,num_uns)
diff_file_path = data_dir + '/diff1/combined_images_diff1_annotations_with_predictions'
diff_relabeled_path = data_dir + '/diff1/combined_ann_diff1_relabeled_th0.9'

relabel_uns_likely_pos(diff_file_path, idx_uns, diff_relabeled_path)

combine_dict_c = {data_dir + '/diff1/combined_images_diff1': diff_relabeled_path}
combine_dict_c[data_dir + '/combined_images_pos'] = data_dir + '/combined_annotations_pos'
combine_dict_c[data_dir + '/combined_images_neg'] = data_dir + '/combined_annotations_neg'
combine_dict_c[data_dir_n + '/neg_combined_images'] = data_dir_n + '/neg_combined_annotations'

combine_datasets(combine_dict_c, data_dir + '/class1/class1_images', data_dir + '/class1/class1_annotations')

# classifier 2: 
# using diff 2: uns labeled as pos with probability >0.9 and neg otherwise
# and all pos/neg from pos slides; all neg from neg slides

idx_uns = np.arange(0,num_uns)
diff_file_path = data_dir + '/diff2/combined_images_diff2_annotations_with_predictions'
diff_relabeled_path = data_dir + '/diff2/combined_ann_diff2_relabeled_th0.9'

relabel_uns_likely_pos(diff_file_path, idx_uns, diff_relabeled_path)

combine_dict_c.pop(data_dir + '/diff1/combined_images_diff1')
combine_dict_c[data_dir + '/diff2/combined_images_diff2'] = diff_relabeled_path
combine_datasets(combine_dict_c, data_dir + '/class2/class2_images', data_dir + '/class2/class2_annotations')

# classifier 3: 
# using diff 3: uns labeled as pos with probability >0.9 and neg otherwise
# and all pos/neg from pos slides; all neg from neg slides

idx_uns = np.arange(0,num_uns)
diff_file_path = data_dir + '/diff3/combined_images_diff3_annotations_with_predictions'
diff_relabeled_path = data_dir + '/diff3/combined_ann_diff3_relabeled_th0.9'

relabel_uns_likely_pos(diff_file_path, idx_uns, diff_relabeled_path)

combine_dict_c.pop(data_dir + '/diff2/combined_images_diff2')
combine_dict_c[data_dir + '/diff3/combined_images_diff3'] = diff_relabeled_path
combine_datasets(combine_dict_c, data_dir + '/class3/class3_images', data_dir + '/class3/class3_annotations')

# classifier 4:
# uns labeled as neg
# and all pos/neg from pos slides; all neg from neg slides

combine_dict_c.pop(data_dir + '/diff3/combined_images_diff3')
combine_dict_c[data_dir + 'combined_images_uns'] = data_dir + 'combined_annotations_uns_labeled_neg'
combine_datasets(combine_dict_c, data_dir + '/class4/class4_images', data_dir + '/class4/class4_annotations')



