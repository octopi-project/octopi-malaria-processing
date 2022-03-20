import numpy as np
import pandas as pd
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
import FlowCal
import os
import pickle
from utils_gating import *
from utils_visualization import *
import gcsfs
import argparse

# for multiprocessing
def process_column(j,spot_data_pd,gcs_settings,dataset_id,parameters,settings):
  bucket_source = settings['bucket_source']
  bucket_destination = settings['bucket_destination']
  fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
  for i in range(parameters['row_start'],parameters['row_end']):
    for k in range(parameters['z_start'],parameters['z_end']):
      file_id = str(i) + '_' + str(j) + '_' + str(k)
      print('processing fov ' + file_id)
      idx = (spot_data_pd['FOV_row']==i) & (spot_data_pd['FOV_col']==j)
      spot_data_current_fov = spot_data_pd[idx]
      print(spot_data_current_fov)
      # read images
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
      process_fov_for_spot_visualizations(I_fluorescence,I_DPC,spot_data_current_fov,settings,parameters,dir_out='test',r=30,scale=5,image_format='jpeg')




      '''
      spot_data_pd = spot_data_pd[spot_data_pd['FOV_row']==i & spot_data_pd['FOV_col']==j]
      print(spot_data_pd)
      '''
      '''
      I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
      # crop image
      I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
      # cv2.imwrite(str(i)+'_'+str(j)+'_'+str(k)+'fluorescence_raw.bmp',cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR))
      # remove background
      I_fluorescence_bg_removed = remove_background(I_fluorescence,return_gpu_image=True)
      cv2.imwrite(str(i) + '_' + str(j) + '_' + str(k) + '_fluorescence_background_removed' + '.' + settings['saving_file_format'],cv2.cvtColor(cp.asnumpy(I_fluorescence_bg_removed),cv2.COLOR_RGB2BGR))
      # detect spots
      spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,downsize_factor=settings['spot_detection_downsize_factor']),thresh=settings['spot_detection_threshold'])
      if(len(spot_list)==0):
        return
      spot_list = prune_blobs(spot_list)
      # process spots
      spot_list, spot_data_pd = process_spots(I_fluorescence_bg_removed,I_fluorescence,spot_list,i,j,k,settings)
      # save image with spot boxed
      if settings['save_spot_detection_visualization']:
        I_boxed = cv2.cvtColor(cp.asnumpy(255*highlight_spots(I_fluorescence_bg_removed,spot_list*settings['spot_detection_downsize_factor'])).astype('uint8'),cv2.COLOR_RGB2BGR)
        with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_detection_result/' + file_id + '.jpg', 'wb' ) as f:
          f.write(cv2.imencode('.jpg',I_boxed)[1].tobytes())
        # cv2.imwrite(str(i) + '_' + str(j) + '_' + str(k) + '_spot_detection_result' + '.' + settings['saving_file_format'],cv2.cvtColor(cp.asnumpy(255*highlight_spots(I_fluorescence_bg_removed,spot_list_pruned*settings['spot_detection_downsize_factor'])).astype('uint8'),cv2.COLOR_RGB2BGR))
      # save the spot list
      with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_lists/' + file_id + '.csv', 'wb' ) as f:
        np.savetxt(f,spot_list,fmt=('%d','%d','%.1f'),delimiter=',')
      # save spot data
      with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data/' + file_id + '.csv', 'wb' ) as f:
        spot_data_pd.to_csv(f,index=False)
      '''