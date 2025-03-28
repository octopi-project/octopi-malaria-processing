import torch
import model_training_utils
import pandas as pd
import numpy as np

# input paths
ims_path = 'example_im.npy' # path to images for training
ann_path = 'example_ann.csv' # path to their respective annotations
model_path = 'example_model.pt' # path to the trained model

# model training parameters
# dictionary for annotations (name:number)
ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}

# output paths
ann_w_pred_path = 'example_ann_w_predictions.csv' # saves the annotations csv but with prediction scores for all classes


# open the input files
ann_df = pd.read_csv(ann_path, index_col='index')
images = np.load(ims_path)
# load the model
if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path,map_location=torch.device('cpu'))

# run the model
model_training_utils.run_model(ann_dict, model, images, ann_df, ann_w_pred_path)