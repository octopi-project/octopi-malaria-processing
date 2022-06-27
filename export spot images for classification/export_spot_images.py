import numpy as np
import pandas as pd
import os
import gcsfs
import argparse
import json
from functools import partial
import multiprocessing as mp
from multiprocessing import get_context
from export_spot_images_column import process_column

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    export_selected_spots = False # selected spots vs all spots
    debug_mode = True
    save_to_gcs = False

    if args.data_id != None:
        DATASET_ID = [args.data_id]
    else:
        f = open('list of datasets_negative.txt','r')
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
        f.close()

    # gcs setting
    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
    bucket_destination = 'gs://octopi-malaria-data-processing'

    settings = {}
    settings['export selected spots'] = export_selected_spots
    settings['bucket_source'] = bucket_source
    settings['bucket_destination'] = bucket_destination
    settings['save to gcs'] = save_to_gcs

    for dataset_id in DATASET_ID:

        print('<processing ' + dataset_id + '>')
    
        settings['dataset_id'] = dataset_id
        
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

        # get spot data
        if export_selected_spots:
            spot_data_pd = pd.read_csv('spot_data_selected_' + dataset_id + '.csv', index_col=None, header=0)
        else:
            with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data_raw.csv', 'r' ) as f:
                spot_data_pd = pd.read_csv(f, index_col=None, header=0)
        
        # process FOV
        with get_context("spawn").Pool(processes=8) as pool:
            pool.map(partial(process_column,spot_data_pd=spot_data_pd,gcs_settings=gcs_settings,dataset_id=dataset_id,parameters=parameters,settings=settings),columns)
