import model_training_utils

# input paths
ims_path = 'example_im.npy' # path to images for training
ann_path = 'example_ann.csv' # path to their respective annotations

# model training parameters
# dictionary for annotations (name:number)
ann_dict = {'non-parasite':0, 'parasite':1, 'unsure':2, 'unlabeled':-1}
# model architecture
model_specs = {'model_name':'resnet34','n_channels':4,'n_filters':64,'n_classes':len(ann_dict)-1,'kernel_size':3,'stride':1,'padding':1, 'batch_size':32}

# output paths
ann_w_pred_path = 'example_ann_w_predictions.csv' # saves the annotations csv but with prediction scores for all classes
model_path = 'example_model.pt' # saves the trained model

model_training_utils.model_training(ann_dict, ims_path, ann_path, ann_w_pred_path, model_path, model_specs) #TODO: PUT BACK!
