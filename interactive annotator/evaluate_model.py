# model_path = ''
# image_path = ''
# annotation_path = 'test' 
# dir_in = ''

batch_size_inference = 2048
target_false_negative_rate = 0.1
target_false_positive = 5
th = 0.96

dir_in = ''
model_path = ''
annotation_path = ''
image_path = ''

# load configurations
##########################################################################
import glob
config_files = glob.glob('.' + '/' + 'configurations*.txt')
if config_files:
    if len(config_files) > 1:
        print('multiple configuration files found, the program will exit')
        exit()
    exec(open(config_files[0]).read())
###########################################################################
import utils
import utils_visualization
import models
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load model
if torch.cuda.is_available():
    loaded_model = torch.load(dir_in + model_path + '.pt')
else:
    loaded_model = torch.load(dir_in + model_path + '.pt',map_location=torch.device('cpu'))
if isinstance(loaded_model, dict):
    model = models.ResNet(model=model_spec['model'],n_channels=model_spec['n_channels'],n_filters=model_spec['n_filters'],
        n_classes=model_spec['n_classes'],kernel_size=model_spec['kernel_size'],stride=model_spec['stride'],padding=model_spec['padding'])
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(dir_in + model_path + '.pt'))
    else:
        model.load_state_dict(torch.load(dir_in + model_path + '.pt',map_location=torch.device('cpu')))
else:
    if torch.cuda.is_available():
        model = torch.load(dir_in + model_path + '.pt')
    else:
        model = torch.load(dir_in + model_path + '.pt',map_location=torch.device('cpu'))

# load images
images = np.load(dir_in + image_path + '.npy')

# load annotations
annotation_pd = pd.read_csv(dir_in + annotation_path + '.csv',index_col='index')

# prepare the data
annotation_pd = annotation_pd[annotation_pd['annotation'].isin([0, 1])]
annotations = annotation_pd['annotation'].values
indices = annotation_pd.index.to_numpy()
images = images[indices,]

# run the model
predictions, features = utils.generate_predictions_and_features(model,images,batch_size_inference)
predictions = predictions.squeeze()

# get the predictions for the positive and negative images
predictions_positive = predictions[annotations==1]
predictions_negative = predictions[annotations==0]
positive_total = len(predictions_positive)
negative_total = len(predictions_negative)

# generate curve
thresholds = np.arange(0,1,0.0025)
n = len(thresholds)
false_negative = np.zeros((n,1),dtype=float)
false_positive = np.zeros((n,1),dtype=float)
for i in range(n):
    false_negative[i] = false_negative[i] + np.sum(predictions_positive <= thresholds[i])
    false_positive[i] = false_positive[i] + np.sum(predictions_negative >= thresholds[i])
# find the number of false positives at false negative rate of 0.2
idx_target_false_negative_rate = np.argmin(abs((false_negative/positive_total)-target_false_negative_rate))
idx_target_false_positive = np.argmin(abs(false_positive-target_false_positive))

# plot the result
fig, ax1 = plt.subplots()
line1 = ax1.plot(thresholds,false_positive)
plt.scatter(thresholds[idx_target_false_positive],false_positive[idx_target_false_positive])
plt.scatter(thresholds[idx_target_false_negative_rate],false_positive[idx_target_false_negative_rate],facecolors='none',edgecolor='C0')
plt.plot([0,thresholds[idx_target_false_negative_rate]],[false_positive[idx_target_false_negative_rate],false_positive[idx_target_false_negative_rate]],':',color='C0')
plt.ylim((1,negative_total))
plt.yscale('log')
plt.grid(True)
ax1.set_ylabel('Fasle Positives', color='C0')
ax1.tick_params(axis='y', color='C0', labelcolor='C0')

ax2 = ax1.twinx()
line2 = ax2.plot(thresholds,false_negative/positive_total,'C1')
ax2.set_ylabel('False Negative Rate', color='C1')
ax2.tick_params(axis='y', color='C1', labelcolor='C1')
ax2.spines['right'].set_color('C1')
ax2.spines['left'].set_color('C0')
plt.ylim((0,1)) 
plt.xlim((0,1)) 
plt.scatter(thresholds[idx_target_false_negative_rate],false_negative[idx_target_false_negative_rate]/positive_total,color='C1')
plt.scatter(thresholds[idx_target_false_positive],false_negative[idx_target_false_positive]/positive_total,facecolors='none',edgecolor='C1')
plt.plot([thresholds[idx_target_false_positive],1],[false_negative[idx_target_false_positive]/positive_total,false_negative[idx_target_false_positive]/positive_total],':',color='C1')
plt.savefig(dir_in + model_path + '_' + image_path + '.png', dpi='figure')

# show false positives
idx = np.squeeze(predictions >= th) & np.squeeze(annotations==0)
utils_visualization.make_movie(images[idx],dir_in + image_path + '_' + model_path + '_' + str(th), scale_factor=5, fps=5)
utils_visualization.save_images(images[idx],dir_in + image_path + '_' + model_path + '_' + str(th), indices[idx], predictions[idx], scale_factor=5)