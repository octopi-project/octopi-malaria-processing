import numpy as np
import pandas as pd
import os
import pickle
from utils_export import *
import gcsfs
import argparse

# for multiprocessing
def process_column(j,spot_data_pd,gcs_settings,dataset_id,parameters,settings,fs=None):
  bucket_source = settings['bucket_source']
  bucket_destination = settings['bucket_destination']
  if fs == None:
    fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
  for i in range(parameters['row_start'],parameters['row_end']):
    for k in range(parameters['z_start'],parameters['z_end']):
      file_id = str(i) + '_' + str(j) + '_' + str(k)
      print('processing fov ' + file_id)
      idx = (spot_data_pd['FOV_row']==i) & (spot_data_pd['FOV_col']==j)
      spot_data_current_fov = spot_data_pd[idx]
      print(spot_data_current_fov.shape[0])
      if spot_data_current_fov.shape[0] > 0:      # read images
        I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
        I_fluorescence = cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR)
        I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
        I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
        # convert to mono if color
        if len(I_BF_left.shape)==3:
          I_BF_left = I_BF_left[:,:,1]
          I_BF_right = I_BF_right[:,:,1]
        # make images float and 0-1
        I_fluorescence = I_fluorescence.astype('float')/255
        I_BF_left = I_BF_left.astype('float')/255
        I_BF_right = I_BF_right.astype('float')/255
        # crop image
        I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
        I_BF_left = I_BF_left[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
        I_BF_right = I_BF_right[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
        # generate dpc
        I_DPC = generate_dpc(I_BF_left,I_BF_right)
        # cv2.imwrite('dpc.png',I_DPC)
        # generate_visualizations
        if settings['save to gcs']:
          dir_out = None
        else:
          dir_out = '/media/octopi/Extreme SSD/Rinni/Octopi/data/all_patient_slides/spot images_' + dataset_id
          if not os.path.exists(dir_out):
            os.mkdir(dir_out)
        export_spot_images_from_fov(I_fluorescence,I_DPC,spot_data_current_fov,parameters,settings,gcs_settings,dir_out=dir_out,r=15)