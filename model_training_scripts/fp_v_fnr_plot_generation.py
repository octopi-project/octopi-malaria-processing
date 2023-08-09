import torch
import model_training_utils
import model_performance_utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# USER PARAMETERS 
dataset_id = 'example' # can include data directory (eg. it can be '/folder/to/dir/example_path_header')

# input paths
ims_path = dataset_id + '_im.npy' # path to images
ann_path = dataset_id + '_ann.csv' # path to their annotations
model_path = dataset_id + '_model.pt' # path to the trained model

# USER PARAMETERS (optional)
unsure_ignored = True 

# intermediate / output paths
out_ann_w_pred_path = dataset_id + '_ann_w_pred.csv' # will save annotations with predictions
out_model_performance_path = dataset_id + '_model_performance.csv' # will save all performance metrics given annotations with predictions
out_fp_v_fnr_path = dataset_id + '_fp_v_fnr.png' # will save plot of FP v FNR


# 1. RUN INFERENCE
# model training parameters
# dictionary for annotations (name:number)
ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}

# open the input files
ann_df = pd.read_csv(ann_path, index_col='index')
images = np.load(ims_path)
# load the model
if torch.cuda.is_available():
    model = torch.load(model_path)
else:
    model = torch.load(model_path,map_location=torch.device('cpu'))

# run the model
model_training_utils.run_model(ann_dict, model, images, ann_df, out_ann_w_pred_path)


# 2. EVALUATE PERFORMANCE
pos_class = 'parasite'

model_performance_utils.model_perf_analysis(out_ann_w_pred_path, out_model_performance_path, pos_class, unsure_ignored)

# 3. GENERATE PLOT

# trends to plot: FP and FNR
trend_1 = 'FP'
trend_2 = 'FNR'

model_performance_utils.model_perf_visualization(dataset_id, out_model_performance_path, out_fp_v_fnr_path, trend_1, trend_2, pos_class)


