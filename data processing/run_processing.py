import os
import gcsfs
import cv2
import time
import json
from functools import partial
import multiprocessing as mp
from utils import *
from process_column import process_column

if __name__ == '__main__':

  # gcs setting
  gcs_project = 'soe-octopi'
  gcs_token = 'data-20220317-keys.json'
  gcs_settings = {}
  gcs_settings['gcs_project'] = gcs_project
  gcs_settings['gcs_token'] = gcs_token

  # dataset ID
  bucket = 'gs://octopi-malaria-tanzania-2021-data'
  dataset_id = 'U3D_201910_2022-01-11_23-11-36.799392'

  # ROI definition
  debug_mode = True

  # other settings
  a = 2800
  k = 0 # z plane
  saving_file_format = 'bmp'
  saving_location = 'local' 
  # saving_location = 'cloud' 

  # deterimine the size of the scan
  fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
  json_file = fs.cat(bucket + '/' + dataset_id + '/acquisition parameters.json')
  acquisition_parameters = json.loads(json_file)

  # regions to process + other settings
  parameters = {}
  parameters['row_start'] = 0
  parameters['row_end'] = acquisition_parameters['Ny']
  parameters['column_start'] = 0
  parameters['column_end'] = acquisition_parameters['Nx']
  parameters['z_start'] = 0
  parameters['z_end'] = acquisition_parameters['Nz']
  if debug_mode:
    parameters['row_end'] = 2
    parameters['column_end'] = 2
  columns = range(parameters['column_start'], parameters['column_end'])

  # processing
  print('processing ' + dataset_id + ' in ' + bucket)
  with mp.Pool(processes=4) as pool:
    pool.map(partial(process_column,gcs_settings=gcs_settings,bucket=bucket,dataset_id=dataset_id,parameters=parameters),columns)