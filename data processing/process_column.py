import os
import gcsfs
import cv2
import time
import json
import multiprocessing as mp
import cupy as cp # conda install -c conda-forge cupy==10.2
from utils import *

# processing implementation
def process_column(j,gcs_settings,dataset_id,parameters,settings):
  bucket_source = settings['bucket_source']
  bucket_destination = settings['bucket_destination']
  fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])
  for i in range(parameters['row_start'],parameters['row_end']):
    for k in range(parameters['z_start'],parameters['z_end']):
      file_id = str(i) + '_' + str(j) + '_' + str(k)
      print('processing fov ' + file_id)
      I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
      # crop image
      I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
      # cv2.imwrite(str(i)+'_'+str(j)+'_'+str(k)+'fluorescence_raw.bmp',cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR))
      # remove background
      I_fluorescence_bg_removed = remove_background(I_fluorescence,return_gpu_image=True)
      cv2.imwrite(str(i) + '_' + str(j) + '_' + str(k) + '_fluorescence_background_removed' + '.' + settings['saving_file_format'],cv2.cvtColor(cp.asnumpy(I_fluorescence_bg_removed),cv2.COLOR_RGB2BGR))
      # detect spots
      spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,downsize_factor=settings['spot_detection_downsize_factor']),thresh=settings['spot_detection_threshold'])
      spot_list_pruned = prune_blobs(spot_list)
      # save image with spot boxed
      if settings['save_spot_detection_visualization']:
        I_boxed = cv2.cvtColor(cp.asnumpy(255*highlight_spots(I_fluorescence_bg_removed,spot_list_pruned*settings['spot_detection_downsize_factor'])).astype('uint8'),cv2.COLOR_RGB2BGR)
        with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_detection_result/' + file_id + '.jpg', 'wb' ) as f:
          f.write(cv2.imencode('.jpg',I_boxed)[1].tobytes())
        # cv2.imwrite(str(i) + '_' + str(j) + '_' + str(k) + '_spot_detection_result' + '.' + settings['saving_file_format'],cv2.cvtColor(cp.asnumpy(255*highlight_spots(I_fluorescence_bg_removed,spot_list_pruned*settings['spot_detection_downsize_factor'])).astype('uint8'),cv2.COLOR_RGB2BGR))
      # save the spot list
      with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_lists/' + file_id + '.csv', 'wb' ) as f:
        np.savetxt(f,spot_list_pruned,fmt=('%d','%d','%.1f'),delimiter=',')