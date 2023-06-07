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

# get images to run predictions on from lists of datasets that were just processed
# f = open('/home/rinni/octopi-malaria/export spot images rb/pos list of datasets.txt','r')
f = open('/home/rinni/octopi-malaria/export spot images rb/neg list of datasets.txt','r')
DATASET_ID = f.read()
DATASET_ID = DATASET_ID.split('\n')
DATASET_ID = DATASET_ID[:-1] # remove empty string at end
f.close()

# dir_in = '/media/rinni/Extreme SSD/Rinni/Octopi/data/pos_testing_slides/'
dir_in = '/media/rinni/Extreme SSD/Rinni/Octopi/data/neg_testing_slides/'

image_paths = [dir_in + x + '.npy' for x in DATASET_ID]

# ann_slide = 1 # 1 if pos, 0 if neg
ann_slide = 0

data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/'
folders = ['s_3a','s_3b']
model_tail = '_r34_b32'

# go through images for predictions
for image_path in image_paths:
    print(image_path)
    images = np.load(image_path)
    ann_df = pd.DataFrame({'index': range(images.shape[0]),'annotation': [ann_slide]*images.shape[0]})

    for folder in folders:
        if folder == 's_a':
            ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2}
        else:
            ann_dict = {'non-parasite':0, 'parasite':1}
        if len(folder) == 3:
            model_name = '/model_perf' + model_tail + '.pt'
        else:
            model_name = '/model_perf_cl' + model_tail + '.pt'

        model = torch.load(data_dir + folder + model_name)

        summary_path = os.path.join(dir_in, folder, 'model' + model_tail + '_summary.txt')
        print(summary_path)
        with open(summary_path, 'w') as f:
            sys.stdout = f
            summary_str = summary(model, input_size=(4,31,31))
        sys.stdout = sys.__stdout__

        out_ann_w_pred_path = dir_in + folder + '/' + os.path.splitext(os.path.basename(image_path))[0] + '_ann_with_predictions'  + model_tail + '.csv'
        print(out_ann_w_pred_path)
        run_model(ann_dict, model, images, ann_df, out_ann_w_pred_path)