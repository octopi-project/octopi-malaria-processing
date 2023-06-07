import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import models
import time
from torchsummary import summary
import os
import sys

# FUNCTIONS
# runs the model and saves the annotations df to include prediction scores
def run_model(ann_dict, model, images, ann_df, out_ann_w_pred_path, batch_size_inference=2048):
    predictions, features = generate_predictions_and_features(model,images,batch_size_inference)

    # make dataframe for outputs
    output_pd = pd.DataFrame(index = np.arange(images.shape[0]))
    i = 0 # counter
    for key in ann_dict:
        if ann_dict[key] >= 0:
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

# GLOBAL VARIABLES
classifier = '/s_3a/'
model_arch = '_r34_b32'
if 's_a' in classifier:
    ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2}
else:
    ann_dict = {'non-parasite':0, 'parasite':1}

im_folder = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/'
model_folder = '/media/rinni/Extreme SSD/Rinni/to-combine/' + classifier

model_path = model_folder + 'model_perf_cl' + model_arch + '.pt'
im_test_path = im_folder + 'combined_images_val.npy'
im_train_path = im_folder + 'combined_images.npy'
val_ind_path = model_folder + 'indices' + model_arch + '.csv'
ann_test_path = im_folder + 'combined_ann_val.csv'
ann_train_path = im_folder + 'combined_ann.csv'

model = torch.load(model_path)
im_test = np.load(im_test_path)
im_train = np.load(im_train_path)
val_indices = np.genfromtxt(val_ind_path, delimiter=',')
val_indices = val_indices.astype(int)
im_val = im_train[val_indices,:,:,:]
ann_test = pd.read_csv(ann_test_path, index_col='index').values.squeeze()
ann_train = pd.read_csv(ann_train_path, index_col='index').values.squeeze()
ann_val = ann_train[val_indices]

im_for_evaluating = np.concatenate((im_test, im_val), axis=0)
print(im_for_evaluating.shape)
ann_for_evaluating = np.concatenate((ann_test, ann_val), axis=0)
ann_for_evaluating = pd.DataFrame({'annotation':ann_for_evaluating})
print(ann_for_evaluating)
ann_for_evaluating.index.name = 'index'

summary_path = model_folder + 'model' + model_arch + '_summary.txt'
print(summary_path)
with open(summary_path, 'w') as f:
    sys.stdout = f
    summary_str = summary(model, input_size=(4,31,31))
sys.stdout = sys.__stdout__

out_ann_w_pred_path = model_folder + '/' + 'model' + model_arch + '_evaluation_ann_with_predictions.csv'
print(out_ann_w_pred_path)
run_model(ann_dict, model, im_for_evaluating, ann_for_evaluating, out_ann_w_pred_path)