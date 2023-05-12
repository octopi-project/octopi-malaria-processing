import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import models
import time
from torchsummary import summary

# GLOBAL VARIABLES

# import images & annotations
# run this through model
# save to given ann with predictions path
ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}

images_path = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_for_training_performance.npy' # all images except patient negatives
ann_path = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_for_training_performance.csv'

images = np.load(images_path)
ann_df = pd.read_csv(ann_path, index_col='index')

data_dir = '/media/rinni/Extreme SSD/Rinni/to-combine/'
folders = ['s_a','s_b','s_1a','s_1b','s_2a','s_2b','s_3a','s_3b']

# runs the model and saves the annotations df to include prediction scores
def run_model(ann_dict, model, images, ann_df, out_ann_w_pred_path, batch_size_inference=2048):
    print(images.shape)
    summary(model, input_size=(4,31,31))
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

for folder in folders:
    if folder == 's_a':
        ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2}
    else:
        ann_dict = {'non-parasite':0, 'parasite':1}
    if len(folder) == 3:
        model_name = '/model_perf_r18_b32.pt'
    else:
        model_name = '/model_perf_cl_r18_b32.pt'
    model = torch.load(data_dir + folder + model_name)
    out_ann_w_pred_path = data_dir + folder + '/ann_with_predictions' + model_name[model_name.index('_r'):model_name.index('.')] + '_new.csv'
    print(out_ann_w_pred_path)
    run_model(ann_dict, model, images, ann_df, out_ann_w_pred_path)