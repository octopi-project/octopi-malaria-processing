import numpy as np
import pandas as pd
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib.pyplot as plt
import FlowCal
import os
import pickle
from utils_gating import *
import gcsfs
import argparse
import json
from functools import partial
import multiprocessing as mp
from multiprocessing import get_context
from generate_spot_visualizations_column import process_column

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    debug_mode = True

    if args.data_id != None:
        dataset_id = args.data_id
    else:
        dataset_id = 'U3D_201910_2022-01-11_23-11-36.799392'

    # gcs setting
    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    # dataset ID
    bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
    bucket_destination = 'gs://octopi-malaria-data-processing'

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
        parameters['row_end'] = 1
        parameters['column_end'] = 1
    columns = range(parameters['column_start'], parameters['column_end'])
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900

    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'

    # dataset ID
    bucket_destination = 'gs://octopi-malaria-data-processing'

    spot_data_pd = pd.read_csv('spot_data_selected_' + dataset_id + '.csv', index_col=None, header=0)
    with get_context("spawn").Pool(processes=8) as pool:
        pool.map(partial(process_column,spot_data_pd=spot_data_pd,gcs_settings=gcs_settings,dataset_id=dataset_id,parameters=parameters,settings=settings),columns)
