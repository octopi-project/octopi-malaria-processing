import os
import gcsfs
import cv2
import time
import json
from functools import partial
import multiprocessing as mp
from utils import *
from process_column import process_column
from multiprocessing import get_context

if __name__ == '__main__':

  # gcs setting
  gcs_project = 'soe-octopi'
  gcs_token = 'data-20220317-keys.json'
  gcs_settings = {}
  gcs_settings['gcs_project'] = gcs_project
  gcs_settings['gcs_token'] = gcs_token

  # dataset ID
  bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
  bucket_destination = 'gs://octopi-malaria-data-processing'
  dataset_id = 'U3D_201910_2022-01-11_23-11-36.799392'

  # ROI definition
  debug_mode = False

  # other settings
  settings = {}
  settings['spot_detection_downsize_factor'] = 4
  settings['saving_file_format'] = 'bmp'
  settings['saving_location'] = 'local'
  settings['downsize_factor'] = 4
  settings['spot_detection_threshold'] = 10
  settings['bucket_source'] = bucket_source
  settings['bucket_destination'] = bucket_destination
  settings['save_spot_detection_visualization'] = True

  # deterimine the size of the scan
  fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
  json_file = fs.cat(bucket_source + '/' + dataset_id + '/acquisition parameters.json')
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
  parameters['crop_x0'] = 100
  parameters['crop_x1'] = 2900
  parameters['crop_y0'] = 100
  parameters['crop_y1'] = 2900

  # processing
  print('processing ' + dataset_id + ' in ' + bucket_source)
  with get_context("spawn").Pool(processes=8) as pool:
    pool.map(partial(process_column,gcs_settings=gcs_settings,dataset_id=dataset_id,parameters=parameters,settings=settings),columns)