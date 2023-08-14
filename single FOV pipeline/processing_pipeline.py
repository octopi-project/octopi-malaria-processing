import os
import cv2
import time
import imageio
from utils import *
from model_utils import *
import matplotlib.pyplot as plt

settings = {}
settings['spot_detection_downsize_factor'] = 4
settings['spot_detection_threshold'] = 10

# regions to process + other settings
parameters = {}
parameters['crop_x0'] = 100
parameters['crop_x1'] = 2900
parameters['crop_y0'] = 100
parameters['crop_y1'] = 2900

def process_fov(I_fluorescence,I_BF_left,I_BF_right,model,device,classification_th):

    # crop image
    I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
    I_BF_left = I_BF_left[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
    I_BF_right = I_BF_right[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]

    # remove background
    I_fluorescence_bg_removed = remove_background(I_fluorescence,return_gpu_image=True)

    # detect spots
    spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,downsize_factor=settings['spot_detection_downsize_factor']),thresh=settings['spot_detection_threshold'])
    if(len(spot_list)==0):
        print('no spots!')
        return None
    spot_list = prune_blobs(spot_list)

    # scale coordinates for full-res image
    spot_list = spot_list*settings['spot_detection_downsize_factor']

    # generate spot arrays
    I_BF_left = I_BF_left.astype('float')/255
    I_BF_right = I_BF_right.astype('float')/255
    I_DPC = generate_dpc(I_BF_left,I_BF_right)
    I_fluorescence = I_fluorescence.astype('float')/255
    I = get_spot_images_from_fov(I_fluorescence,I_DPC,spot_list,r=15)
    I = I.transpose(0, 3, 1, 2)

    # classify
    prediction_score_negative_sample = run_model(model,device,I)[:,1]
    indices = np.where(prediction_score_negative_sample > classification_th)[0]

    # return positive spots
    return I[indices]