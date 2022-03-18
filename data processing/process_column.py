import os
import gcsfs
import cv2
import time
import json
import multiprocessing as mp
from utils import *

# processing implementation
def process_column(j,gcs_settings,bucket,dataset_id,parameters):
  fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
  for i in range(parameters['row_start'],parameters['row_end']):
    for k in range(parameters['z_start'],parameters['z_end']):
      file_id = str(i) + '_' + str(j) + '_' + str(k)
      print('processing fov ' + file_id)
      I_fluorescence = imread_gcsfs(fs,bucket + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
      # crop image
      I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
      # cv2.imwrite(str(i)+'_'+str(j)+'_'+str(k)+'fluorescence_raw.bmp',cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR))
      # remove background
      I_fluorescence_bg_removed = remove_background(I_fluorescence,return_gpu_image=False)
      cv2.imwrite(str(i)+'_'+str(j)+'_'+str(k)+'_fluorescence_background_removed.bmp',cv2.cvtColor(I_fluorescence_bg_removed,cv2.COLOR_RGB2BGR)