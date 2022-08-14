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
import FlowCal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

  # gcs setting
  gcs_project = 'soe-octopi'
  gcs_token = 'data-20220317-keys.json'
  gcs_settings = {}
  gcs_settings['gcs_project'] = gcs_project
  gcs_settings['gcs_token'] = gcs_token

  # dataset ID
  # bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
  # bucket_destination = 'gs://octopi-malaria-data-processing'
  bucket_source = 'gs://octopi-malaria-uganda-2022-data'
  bucket_destination = 'gs://octopi-malaria-uganda-2022-data-processing'

  # ROI definition
  debug_mode = False

  # pooling and plotting
  pooling_and_plotting = True

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

  f = open('list of datasets.txt','r')
  DATASET_ID = f.read()
  DATASET_ID = DATASET_ID.split('\n')
  f.close()

  # go through dataset
  for dataset_id in DATASET_ID:

    # processing
    print('processing ' + dataset_id + ' in ' + bucket_source)
    with get_context("spawn").Pool(processes=8) as pool:
      pool.map(partial(process_column,gcs_settings=gcs_settings,dataset_id=dataset_id,parameters=parameters,settings=settings),columns)

    if pooling_and_plotting:
      # pool spot data from all fovs
      spot_data_pd = pd.DataFrame()
      for i in range(parameters['row_start'],parameters['row_end']):
        for j in range(parameters['column_start'],parameters['column_end']):
          for k in range(parameters['z_start'],parameters['z_end']):
            file_id = str(i) + '_' + str(j) + '_' + str(k)
            if fs.exists(bucket_destination + '/' + dataset_id + '/' + 'spot_data/' + file_id + '.csv'):
              with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data/' + file_id + '.csv', 'r' ) as f:
                print(file_id)
                spot_data_fov = pd.read_csv(f, index_col=None, header=0)
                print(spot_data_fov)
                spot_data_pd = pd.concat([spot_data_pd,spot_data_fov])
      with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data_raw.csv', 'wb' ) as f:
        spot_data_pd.to_csv(f,index=False)

      # generate scatter plot
      # moved spots with saturated pixels
      idx_spot_with_saturated_pixels = spot_data_pd['numSaturatedPixels']>0
      spot_data_pd = spot_data_pd[~idx_spot_with_saturated_pixels]
      # get RGB
      R = spot_data_pd['R'].to_numpy()
      G = spot_data_pd['G'].to_numpy()
      B = spot_data_pd['B'].to_numpy()
      s = np.vstack((R/B,G/B)).T
      FlowCal.plot.density2d(s, mode='scatter',xscale='linear',yscale='linear',xlim=[0,0.75],ylim=[0,1.5])
      plt.xlabel("R/B")
      plt.ylabel("G/B")
      with fs.open( bucket_destination + '/' + dataset_id + '/' + 'scatter plot_raw.png', 'wb' ) as f:
        plt.savefig(f)
      print(R.size)