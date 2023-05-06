import numpy as np
import pandas as pd
import models
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torch.optim import Adam
import copy
import time

# import utils.py from interactive_annotator
from importlib.machinery import SourceFileLoader
utils = SourceFileLoader("utils", "/home/rinni/octopi-malaria/interactive annotator/utils.py").load_module()


# GLOBAL VARIABLES

ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}
relabel_thresh = 0.9

# folder containing all image and annotation files to filter
data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/'
data_dir_base = data_dir.split('sorted_images')[0]
# folder containing all non-parasite image and annotation files to filter
data_dir_n = '/home/rinni/Desktop/Octopi/data/neg-to-combine/'

# model default parameters
model_spec1 = {'model_name':'resnet18','n_channels':4,'n_filters':64,'n_classes':len(ann_dict)-1,'kernel_size':3,'stride':1,'padding':1, 'batch_size':32}

# model_spec2 = {'model_name':'resnet34','n_channels':4,'n_filters':64,'n_classes':len(ann_dict)-1,'kernel_size':3,'stride':1,'padding':1, 'batch_size':20}

# FUNCTIONS

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

# splits all images/annotations based on their classifications
# opens images/annotations from given paths; splits out the classes given in ann_dict (class name:#)
# saves to default file names, unless out_im_paths/out_ann_paths (dictionaries with class name:file name) are defined
def split_by_class(ann_dict, in_im_path, in_ann_path, out_im_paths = None, out_ann_paths = None):
	images = np.load(in_im_path)
	ann_df = pd.read_csv(in_ann_path,index_col='index')
	for i, class_name in enumerate(ann_dict.keys()):
		if ann_dict[class_name] >= 0:
			idx = ann_df['annotation'] == ann_dict[class_name]
			ann_class = ann_df.loc[idx,'annotation'].values.squeeze()
			ann_split_df = pd.DataFrame({'annotation':ann_class})
			ann_split_df.index.name = 'index'
			images_class = images[idx,]

			# save
			if out_im_paths is None:
				np.save(in_im_path.replace('.npy', '_' + class_name + '.npy'), images_class) # add class_name to file name
			else:
				if class_name in out_im_paths:
					np.save(out_im_paths[class_name], images_class)
				else:
					print('Couldn\'t save image file for class: ' + class_name)
			if out_ann_paths is None:
				ann_split_df.to_csv(in_ann_path.replace('.csv', '_' + class_name + '.csv')) # add class_name to file name
			else:
				if class_name in out_ann_paths:
					ann_split_df.to_csv(out_ann_paths[class_name])
				else:
					print('Couldn\'t save annotation file for class: ' + class_name)

# relabels all images annotated with value og_label (numerical) to have annotation new_label (numerical)
def change_annotation_value(ann_dict, in_ann_path, og_label, new_label, out_ann_path = None):
	ann_df = pd.read_csv(in_ann_path,index_col='index')
	ann_df.loc[ann_df['annotation'] == og_label, 'annotation'] = new_label

	# save
	new_label = round(new_label)
	if out_ann_path is None: # default: {og_class}_as_{new_class}.csv
		out_ann_path = "/".join(in_ann_path.split("/")[:-1])
		out_ann_path += '/combined_ann_unsure_as_' 
		out_ann_path += next((key for key, value in ann_dict.items() if value == round(new_label)), str(round(new_label)))
		out_ann_path += '.csv'
	print('Saving to: ' + out_ann_path)
	ann_df.to_csv(out_ann_path)

# split images from a class into correctly classified vs. misclassified
def isolate_wrong_predictions(ann_dict, class_key, in_im_path, in_ann_w_pred_path, out_dir = None):
	images = np.load(in_im_path)
	ann_df = pd.read_csv(in_ann_w_pred_path,index_col='index')

	# get indices
	cond_ann = ann_df['annotation'] == ann_dict[class_key]
	cond_labeled_w = ann_df[class_key + ' output'] < 0.5
	cond_w = cond_ann & cond_labeled_w
	idx_w = ann_df.loc[cond_w].index

	# get images / annotations
	images_w = images[idx_w,]
	ann_w = ann_df.loc[idx_w,'annotation'].values.squeeze()
	ann_df_w = pd.DataFrame({'annotation':ann_w})
	ann_df_w.index.name = 'index'

	# save
	if out_dir is None: # use the directory for input annotations
		out_dir = '/'.join(in_ann_w_pred_path.split('/')[:-1]) + '/'
	np.save(out_dir + '/combined_images_' + class_key + '_wrong.npy', images_w)
	ann_df_w.to_csv(out_dir + '/combined_ann_' + class_key + '_wrong.csv')	

# train model given input annotations and images; output performance??
def model_training(ann_dict, in_im_path, in_ann_path, out_ann_w_pred_path, out_model_path, model_specs, train_frac=0.7, n_epochs=40):
	# model input prep!
	# load in images and annotations
	images = np.load(in_im_path)
	ann_df = pd.read_csv(in_ann_path, index_col='index')
	# manipulate images/annotations:
	# round annotation labels (recall: they were made non-integers to keep unsure images separate)
	ann_df_round = ann_df.copy()
	ann_df_round['annotation'] = ann_df_round['annotation'].round()
	# remove unlabeled images from dataset
	ann_df_round = ann_df_round[ann_df_round['annotation'].isin([val for val in ann_dict.values() if val >= 0])]
	indices = ann_df_round.index.to_numpy()
	# save annotations
	annotations = ann_df_round['annotation'].values
	images_cut = images[indices,]

	# initialize the model
	print('initialize the model')
	n_classes_derived = ann_df_round['annotation'].nunique() # number of unique annotation classes in dataset
	model = models.ResNet(model=model_specs['model_name'],n_channels=model_specs['n_channels'],n_filters=model_specs['n_filters'], n_classes=n_classes_derived,kernel_size=model_specs['kernel_size'],stride=model_specs['stride'],padding=model_specs['padding'])

	# train model: saves the trained model to out_model_path
	print('train the model')
	train_model(model, images_cut, annotations, out_model_path, train_frac, model_specs['batch_size'], n_epochs)

	# run model; saves the predictions to out_ann_w_pred_path
	batch_size_inference = 2048
	print('load trained model')
	model_new = torch.load(out_model_path)
	print('run model')
	# pass in the annotations df without rounding (so that the saved ann_w_pred aren't rounded and we can identify the former-unsure images)
	run_model(ann_dict, model_new, images, ann_df, out_ann_w_pred_path, batch_size_inference)

# helper function: trains the model given the initialized model, images, and annotations (and other params)
def train_model(model, images, annotations, out_model_path, train_frac=0.7, batch_size=32, n_epochs=40):
	model_best = None

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	# make images 0-1 if they are not already
	if images.dtype == np.uint8:
		images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

	# shuffle
	indices = np.random.choice(len(images), len(images), replace=False)
	data = images[indices,:,:,:]
	label = annotations[indices]

	print('splitting with ' + str(round(train_frac,1)) + ':' + str(round(1-train_frac,1)) + ' train:test split')
	# Split the data into train, validation, and test sets
	X_train, X_val = np.split(data, [int(train_frac * len(data))])
	y_train, y_val = np.split(label, [int(train_frac * len(label))])

	# Create TensorDatasets for train, validation and test
	train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
	val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

	train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=4)
	val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=4)

	# initialize stats
	best_validation_loss = np.inf

	# Define the loss function and optimizer
	# can add weight parameter to differently weight classes; can also add label_smoothing to avoid overfitting
	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=1e-3)

	print('making loss df')
	# initialize loss df
	loss_df = pd.DataFrame(columns=['running loss', 'validation loss'])
	# Training loops
	for epoch in range(n_epochs):
		running_loss = 0.0
		model.train()
		# by batch size?
		for inputs, labels in train_dataloader:
			inputs = inputs.float().to(device)
			labels = labels.float().to(device)
			
			# Forward pass
			outputs = model(inputs)
			labels = labels.to(torch.long)
			loss = criterion(outputs, labels)

			# Backward and optimize
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			running_loss += loss.item()

		# Compute the validation performance
		validation_loss = evaluate_model(model, val_dataloader, criterion, device)
		if validation_loss < best_validation_loss:
			best_validation_loss = validation_loss
			model_best = copy.deepcopy(model)
		# save running_loss and validation_loss
		new_loss = pd.DataFrame({'running loss': running_loss, 'validation loss': validation_loss}, index=[0])
		print('Epoch ' + str(epoch) + ': running loss - '); print(running_loss); print('; validation loss - '); print(validation_loss)
		loss_df = pd.concat([loss_df, new_loss], ignore_index=True)
		
	# training complete
	print('saving the best model to ' + out_model_path)
	torch.save(model_best, out_model_path)
	# TODO: save the best model as well
	# if model_best is not None:
	# 	print('saving the best model to ' + out_model_path.split('.')[0] + '_best_model.' + out_model_path.split('.')[1])
	# 	torch.save(model_best, out_model_path.split('.')[0] + '_best_model.' + out_model_path.split('.')[1])

def evaluate_model(model, dataloader, criterion, device):
	model.eval()

	total_loss = 0.0
	with torch.no_grad():
		for inputs, labels in dataloader:
			inputs = inputs.float().to(device)
			labels = labels.float().to(device)

			# Forward pass
			outputs = model(inputs)
			labels = labels.to(torch.long)
			loss = criterion(outputs, labels)

			total_loss += loss.item()
	return total_loss

# runs the model and saves the annotations df to include prediction scores
def run_model(ann_dict, model, images, ann_df, out_ann_w_pred_path, batch_size_inference=2048):
	predictions, features = generate_predictions_and_features(model,images,batch_size_inference)

	# make dataframe for outputs
	output_pd = pd.DataFrame(index = np.arange(images.shape[0]))
	i = 0 # counter
	for key in ann_dict:
		if ann_dict[key] in np.round(ann_df['annotation'].values) and ann_dict[key] >= 0:
			output_pd[key + ' output'] = predictions[:,i]
			i += 1

	# add it to ann_df
	ann_df = ann_df.filter(regex='^(?!.*output).*$', axis=1) # drop any output columns currently there
	ann_w_pred_df = ann_df.merge(output_pd,left_index=True,right_index=True) # add in new outputs
	ann_w_pred_df = ann_w_pred_df.sort_values('parasite output',ascending=False) # sort by parasite predictions

	# save
	ann_w_pred_df.to_csv(out_ann_w_pred_path)

# gets the predictions (and features?) given a model and input images
def generate_predictions_and_features(model, images, batch_size_inference = 2048):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.to(device)

	if images.dtype == np.uint8:
		images = images.astype(np.float32)/255.0 # convert to 0-1 if uint8 input

	# build dataset
	dataset = TensorDataset(torch.from_numpy(images), torch.from_numpy(np.ones(images.shape[0])))

	# dataloader
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size_inference, shuffle=False)

	# run inference 
	all_features = []
	all_predictions = []
	t0 = time.time()
	for k, (images, labels) in enumerate(dataloader):
		images = images.float().to(device)

		predictions, features = model.get_predictions_and_features(images)
		predictions = predictions.detach().cpu().numpy()
		features = features.detach().cpu().numpy().squeeze()

		all_predictions.append(predictions)
		all_features.append(features)
	predictions = np.vstack(all_predictions)
	features = np.vstack(all_features)
	print('running inference on ' + str(predictions.shape[0]) + ' images took ' + str(time.time()-t0) + ' s')

	return predictions, features

# relabel all unsure images based on their parasite (class with label 1) prediction
# diff here means differentiator
# saves full unsure pd (including non-unsure images)
def relabel_uns_by_diff_predictions(in_ann_w_pred_path, class_val, class_name = 'unsure', out_ann_path = None, relabel_thresh = 0.9, class_to_relabel_by = 'parasite'):
	diff_ann_df = pd.read_csv(in_ann_w_pred_path, index_col='index')
	cond_ann = diff_ann_df['annotation'] == class_val
	cond_pos = diff_ann_df[class_to_relabel_by + ' output'] > relabel_thresh
	cond = cond_ann & cond_pos
	cond_not = cond_ann & ~cond_pos

	# relabel and trim annotation dataframe
	diff_ann_df.loc[cond, 'annotation'] = 1
	diff_ann_df.loc[cond_not, 'annotation'] = 0
	diff_ann_df = diff_ann_df.loc[cond_ann] # only keep rows for images of class_name (unsure)
	diff_ann_df = diff_ann_df.loc[:, ~diff_ann_df.columns.str.contains('output')] # remove all output columns

	# save
	if out_ann_path is None: # default path
		diff_ann_relabeled_path = '/'.join(in_ann_w_pred_path.split('/')[:-1]) + '/' + class_name + '_relabeled_thresh_' + "{:.1f}".format(relabel_thresh) + '.csv'
	diff_ann_df.to_csv(diff_ann_relabeled_path)

# sample strat_name: s_1a
# sample uns_label: 0.8 (what the unsures were relabeled as)
def differentiator_classifier_wrapper(strat_name, uns_label, combine_dict_diff, combine_dict_cl, model_spec = model_spec1):
	# train differentiator
	# combine datasets for differentiator
	diff_images_path = data_dir_base + '/' + strat_name + '/combined_images_diff.npy'
	diff_ann_path = data_dir_base + '/' + strat_name + '/combined_ann_diff.csv'
	combine_datasets(combine_dict_diff, diff_images_path, diff_ann_path)

	# train differentiator 1
	diff_ann_w_pred_path = data_dir_base + '/' + strat_name + '/ann_with_predictions_diff.csv'
	diff_model_perf_path = data_dir_base + '/' + strat_name + '/model_perf_diff.pt'
	model_training(ann_dict, diff_images_path, diff_ann_path, diff_ann_w_pred_path, diff_model_perf_path, model_spec)

	# relabel unsures with differentiator 1
	# save to something like: data_dir + '/s_1a/unsure_relabeled_thresh_0.9.csv'
	relabel_uns_by_diff_predictions(diff_ann_w_pred_path, uns_label)

	# train classifier 1
	# combine datasets for classifier 1
	cl_images_path = data_dir_base + '/' + strat_name + '/combined_images_cl.npy'
	cl_ann_path = data_dir_base + '/' + strat_name + '/combined_ann_cl.csv'
	combine_datasets(combine_dict_cl, cl_images_path, cl_ann_path)

	# train classifier 1
	cl_ann_w_pred_path = data_dir_base + '/' + strat_name + '/ann_with_predictions_cl.csv'
	cl_model_perf_path = data_dir_base + '/' + strat_name + '/model_perf_cl.pt'
	model_training(ann_dict, cl_images_path, cl_ann_path, cl_ann_w_pred_path, cl_model_perf_path, model_spec)


# SET-UP
print('set up!')
# set up images and ann to filter

# combine patient and negative slide images
slide_images_dict = {}
slide_images_dict[data_dir + '/patient_combined_images.npy'] = data_dir + '/patient_combined_ann.csv'
slide_images_dict[data_dir_n + '/neg_combined_images.npy'] = data_dir_n + '/neg_combined_ann.csv'

combined_image_path = data_dir + '/combined_images.npy'
combined_ann_path = data_dir + '/combined_ann.csv'
combine_datasets(slide_images_dict, combined_image_path, combined_ann_path)

# split the combined images by class
split_by_class(ann_dict, combined_image_path, combined_ann_path)
# save combined parasite and non-parasite images
pos_neg_dict = {}
pos_neg_dict[data_dir + '/combined_images_non-parasite.npy'] = data_dir + '/combined_ann_non-parasite.csv'
pos_neg_dict[data_dir + '/combined_images_parasite.npy'] = data_dir + '/combined_ann_parasite.csv'

pos_neg_image_path = data_dir + '/combined_images_parasite_and_non-parasite.npy' 
pos_neg_ann_path = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_datasets(pos_neg_dict, pos_neg_image_path, pos_neg_ann_path)
# save the unsure images, but labeled as various classes (pos, neg, unlabeled)
unsure_ann_path = data_dir + '/combined_ann_unsure.csv'
change_annotation_value(ann_dict, unsure_ann_path, 2, 0.8) # parasite
change_annotation_value(ann_dict, unsure_ann_path, 2, 0.2) # non-parasite
change_annotation_value(ann_dict, unsure_ann_path, 2, -0.8) # unlabeled

'''
# SANITY CHECK; TODO: remove

# set up
# try on 9K images from patient_combined_images/ann; get 3K of each class
ann_path_sanity = data_dir + '/patient_combined_ann.csv'
ann_df_sanity = pd.read_csv(ann_path_sanity, index_col='index')
im_path_sanity = data_dir + '/patient_combined_images.npy'
im_sanity = np.load(im_path_sanity)

# Group the DataFrame by 'annotation' column
grouped = ann_df_sanity.groupby('annotation')

# List to store the indices
indices = []

# Retrieve 3000 random indices for each unique element
for key, group in grouped:
	if key >= 0:
	    unique_element_indices = group.index[:10].tolist()
	    ims = im_sanity[unique_element_indices,]
	    np.save(data_dir_base + '/sanity/' + str(key) + '_im.npy', ims)
	    anns = ann_df_sanity.loc[unique_element_indices,:]
	    anns = anns['annotation'].values.squeeze()
	    anns = pd.DataFrame({'annotation': anns})
	    anns.index.name = 'index'
	    anns.to_csv(data_dir_base + '/sanity/' + str(key) + '_ann.csv')

	    indices.extend(unique_element_indices)

print(indices)
ann_df_sanity = ann_df_sanity.loc[indices,:]
print(ann_df_sanity)
ann_df_sanity = ann_df_sanity['annotation'].values.squeeze()
ann_df_sanity = pd.DataFrame({'annotation':ann_df_sanity})
ann_df_sanity.index.name = 'index'
im_sanity = im_sanity[indices,]

print(ann_df_sanity)

combined_ann_path = data_dir_base + '/sanity/combined_ann.csv'
combined_image_path = data_dir_base + '/sanity/combined_im.npy'
ann_df_sanity.to_csv(combined_ann_path)
np.save(combined_image_path, im_sanity)

# cl_san_ann_w_pred_path = data_dir_base + '/sanity/ann_with_predictions.csv'
# cl_san_model_path = data_dir_base + '/sanity/model_perf.pt'
# model_training(ann_dict, combined_image_path, combined_ann_path, cl_san_ann_w_pred_path, cl_san_model_path, model_spec1)
'''

# FIRST CLASSIFIERS
print('first classifiers!')

# SA: classifier run on all images (in all 3 classes)
cl_sa_ann_w_pred_path = data_dir_base + '/s_a/ann_with_predictions.csv'
cl_sa_model_path = data_dir_base + '/s_a/model_perf.pt'
model_training(ann_dict, combined_image_path, combined_ann_path, cl_sa_ann_w_pred_path, cl_sa_model_path, model_spec1)

# SB: classifier run on all pos/neg images (not unsure)
cl_sb_ann_w_pred_path = data_dir_base + '/s_b/ann_with_predictions.csv'
cl_sb_model_path = data_dir_base + '/s_b/model_perf.pt'
model_training(ann_dict, pos_neg_image_path, pos_neg_ann_path, cl_sb_ann_w_pred_path, cl_sb_model_path, model_spec1)

# "WRONG" PREDICTIONS BY ANNOTATION
print('get wrong predictions!')
# SA: use SA classifier to split out parasite/non-parasite images that are wrong
# saves to data_dir_base + ‘/s_a/combined_images_parasite_wrong.npy’ and data_dir_base + ‘/s_a/combined_ann_parasite_wrong.csv'
isolate_wrong_predictions(ann_dict, 'parasite', combined_image_path, cl_sa_ann_w_pred_path)
# saves to data_dir_base + ‘/s_a/combined_images_non-parasite_wrong.npy’ and data_dir_base + ‘/s_a/combined_ann_non-parasite_wrong.csv'
isolate_wrong_predictions(ann_dict, 'non-parasite', combined_image_path, cl_sa_ann_w_pred_path)

# SB: use SB classifier to split out parasite/non-parasite images that are wrong
# saves to data_dir_base + ‘/s_b/combined_images_parasite_wrong.npy’ and data_dir_base + ‘/s_b/combined_ann_parasite_wrong.csv'
isolate_wrong_predictions(ann_dict, 'parasite', pos_neg_image_path, cl_sb_ann_w_pred_path)
# saves to data_dir_base + ‘/s_b/combined_images_non-parasite_wrong.npy’ and data_dir_base + ‘/s_b/combined_ann_non-parasite_wrong.csv'
isolate_wrong_predictions(ann_dict, 'non-parasite', pos_neg_image_path, cl_sb_ann_w_pred_path)


# START STRATEGIES 1-3: differentiator pipelines
# S1: differentiator uses unsure labeled as parasite and wrong non-parasite
# S1.A: using "wrong" as defined by classifier A
print('S1.A')

# datasets to combine for differentiator 1a
combine_dict_diff = {}
combine_dict_diff[data_dir + '/combined_images_unsure.npy'] = data_dir + '/combined_ann_unsure_as_parasite.csv'
combine_dict_diff[data_dir_base + '/s_a/combined_images_non-parasite_wrong.npy'] = data_dir_base + '/s_a/combined_ann_non-parasite_wrong.csv'

# datasets to combine for classifier 1a: note that the relabeling hasn't actually happened yet!
combine_dict_cl = {}
combine_dict_cl[data_dir + '/combined_images_parasite_and_non-parasite.npy'] = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_dict_cl[data_dir + '/combined_images_unsure.npy'] = data_dir_base + '/s_1a/unsure_relabeled_thresh_0.9.csv'

differentiator_classifier_wrapper('s_1a', 0.8, combine_dict_diff, combine_dict_cl, model_spec1)
print('hi')

# S1.B: using "wrong" as defined by classifier B
print('S1.B')

# datasets to combine for differentiator 1b
combine_dict_diff = {}
combine_dict_diff[data_dir + '/combined_images_unsure.npy'] = data_dir + '/combined_ann_unsure_as_parasite.csv'
combine_dict_diff[data_dir_base + '/s_b/combined_images_non-parasite_wrong.npy'] = data_dir_base + '/s_b/combined_ann_non-parasite_wrong.csv'

# datasets to combine for classifier 1b: note that the relabeling hasn't actually happened yet!
combine_dict_cl = {}
combine_dict_cl[data_dir + '/combined_images_parasite_and_non-parasite.npy'] = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_dict_cl[data_dir + '/combined_images_unsure.npy'] = data_dir_base + '/s_1b/unsure_relabeled_thresh_0.9.csv'

differentiator_classifier_wrapper('s_1b', 0.8, combine_dict_diff, combine_dict_cl, model_spec1)

# S2: differentiator uses unsure labeled as parasite, wrong non-parasite, and wrong parasite
# S2.A: using "wrong" as defined by classifier A
print('S2.A')

# datasets to combine for differentiator 2a
combine_dict_diff = {}
combine_dict_diff[data_dir + '/combined_images_unsure.npy'] = data_dir + '/combined_ann_unsure_as_parasite.csv'
combine_dict_diff[data_dir_base + '/s_a/combined_images_non-parasite_wrong.npy'] = data_dir_base + '/s_a/combined_ann_non-parasite_wrong.csv'
combine_dict_diff[data_dir_base + '/s_a/combined_images_parasite_wrong.npy'] = data_dir_base + '/s_a/combined_ann_parasite_wrong.csv'

# datasets to combine for classifier 2a: note that the relabeling hasn't actually happened yet!
combine_dict_cl = {}
combine_dict_cl[data_dir + '/combined_images_parasite_and_non-parasite.npy'] = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_dict_cl[data_dir + '/combined_images_unsure.npy'] = data_dir_base + '/s_2a/unsure_relabeled_thresh_0.9.csv'

differentiator_classifier_wrapper('s_2a', 0.8, combine_dict_diff, combine_dict_cl, model_spec1)

# S2.B: using "wrong" as defined by classifier A
print('S2.B')

# datasets to combine for differentiator 2b
combine_dict_diff = {}
combine_dict_diff[data_dir + '/combined_images_unsure.npy'] = data_dir + '/combined_ann_unsure_as_parasite.csv'
combine_dict_diff[data_dir_base + '/s_b/combined_images_non-parasite_wrong.npy'] = data_dir_base + '/s_b/combined_ann_non-parasite_wrong.csv'
combine_dict_diff[data_dir_base + '/s_b/combined_images_parasite_wrong.npy'] = data_dir_base + '/s_b/combined_ann_parasite_wrong.csv'

# datasets to combine for classifier 2b: note that the relabeling hasn't actually happened yet!
combine_dict_cl = {}
combine_dict_cl[data_dir + '/combined_images_parasite_and_non-parasite.npy'] = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_dict_cl[data_dir + '/combined_images_unsure.npy'] = data_dir_base + '/s_2b/unsure_relabeled_thresh_0.9.csv'

differentiator_classifier_wrapper('s_2b', 0.8, combine_dict_diff, combine_dict_cl, model_spec1)

# S3: differentiator uses unsure not labeled, wrong non-parasite, and wrong parasite
# S3.A: using "wrong" as defined by classifier A
print('S3.A')

# datasets to combine for differentiator 3a
combine_dict_diff = {}
combine_dict_diff[data_dir + '/combined_images_unsure.npy'] = data_dir + '/combined_ann_unsure_as_unlabeled.csv'
combine_dict_diff[data_dir_base + '/s_a/combined_images_non-parasite_wrong.npy'] = data_dir_base + '/s_a/combined_ann_non-parasite_wrong.csv'
combine_dict_diff[data_dir_base + '/s_a/combined_images_parasite_wrong.npy'] = data_dir_base + '/s_a/combined_ann_parasite_wrong.csv'

# datasets to combine for classifier 3a: note that the relabeling hasn't actually happened yet!
combine_dict_cl = {}
combine_dict_cl[data_dir + '/combined_images_parasite_and_non-parasite.npy'] = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_dict_cl[data_dir + '/combined_images_unsure.npy'] = data_dir_base + '/s_3a/unsure_relabeled_thresh_0.9.csv'

differentiator_classifier_wrapper('s_3a', -0.8, combine_dict_diff, combine_dict_cl, model_spec1)

# S3.B: using "wrong" as defined by classifier B
print('S3.B')

# datasets to combine for differentiator 3b
combine_dict_diff = {}
combine_dict_diff[data_dir + '/combined_images_unsure.npy'] = data_dir + '/combined_ann_unsure_as_unlabeled.csv'
combine_dict_diff[data_dir_base + '/s_b/combined_images_non-parasite_wrong.npy'] = data_dir_base + '/s_b/combined_ann_non-parasite_wrong.csv'
combine_dict_diff[data_dir_base + '/s_b/combined_images_parasite_wrong.npy'] = data_dir_base + '/s_b/combined_ann_parasite_wrong.csv'

# datasets to combine for classifier 3b: note that the relabeling hasn't actually happened yet!
combine_dict_cl = {}
combine_dict_cl[data_dir + '/combined_images_parasite_and_non-parasite.npy'] = data_dir + '/combined_ann_parasite_and_non-parasite.csv'
combine_dict_cl[data_dir + '/combined_images_unsure.npy'] = data_dir_base + '/s_3b/unsure_relabeled_thresh_0.9.csv'

differentiator_classifier_wrapper('s_3b', -0.8, combine_dict_diff, combine_dict_cl, model_spec1)

torch.cuda.empty_cache() # TODO: not sure