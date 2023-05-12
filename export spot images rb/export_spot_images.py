import numpy as np
import pandas as pd
import os
import shutil
import gcsfs
import argparse
import json
import glob
import xarray as xr
import zarr
from functools import partial
import multiprocessing as mp
from multiprocessing import get_context
from export_spot_images_column import process_column

version = 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    export_selected_spots = False # selected spots vs all spots
    save_intermediate_to_gcs = False # spot saved to the local folder: spot images_DATASET ID
    combine_zarr = True

    debug_mode = False
    save_locally = False

    if args.data_id != None:
        DATASET_ID = [args.data_id]
    else:
        f = open('list of datasets.txt','r')
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
        f.close()

    # gcs setting
    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_source = 'octopi-malaria-uganda-2022-data'
    bucket_destination = 'octopi-malaria-uganda-2022-data-processing'

    settings = {}
    settings['export selected spots'] = export_selected_spots
    settings['bucket_source'] = bucket_source
    settings['bucket_destination'] = bucket_destination
    settings['save to gcs'] = save_intermediate_to_gcs

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
            parameters['row_end'] = 2
            parameters['column_end'] = 2
        columns = range(parameters['column_start'], parameters['column_end'])
        parameters['crop_x0'] = 100
        parameters['crop_x1'] = 2900
        parameters['crop_y0'] = 100
        parameters['crop_y1'] = 2900

        '''
        # get spot data
        if export_selected_spots:
            spot_data_pd = pd.read_csv('spot_data_selected_' + dataset_id + '.csv', index_col=None, header=0)
        else:
            with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data_raw.csv', 'r' ) as f:
                spot_data_pd = pd.read_csv(f, index_col=None, header=0)
        
        # process FOV
        print("parallel extraction of spot images from dataset " + dataset_id)
        with get_context("spawn").Pool(processes=8) as pool:
            pool.map(partial(process_column,spot_data_pd=spot_data_pd,gcs_settings=gcs_settings,dataset_id=dataset_id,parameters=parameters,settings=settings),columns)
        '''

        # combine images from different FOV and generate mapping
        print("combine images from different FOVs")
        dir_in = 'spot images_' + dataset_id
        mapping_pd = pd.DataFrame()
        counter = 0
        for i in range(parameters['row_start'],parameters['row_end']):
            for j in range(parameters['column_start'],parameters['column_end']):
                file_id = str(i) + '_' + str(j)
                if os.path.exists(dir_in + '/' + file_id + '.csv'):
                    mapping_fov = pd.read_csv(dir_in + '/' + file_id + '.csv', header=0,index_col=0)
                    mapping_pd = pd.concat([mapping_pd,mapping_fov])
                if os.path.exists(dir_in + '/' + file_id + '.npy'):
                    ds = np.load(dir_in + '/' + file_id + '.npy',allow_pickle=True)
                    if counter == 0:
                        data_all = ds
                    else:
                        data_all = np.concatenate([data_all,ds],axis=0)
                print(counter)
                counter = counter + 1
        mapping_pd.reset_index(inplace=True)
        mapping_pd.rename(columns={'index':'global_index'},inplace=True)
        mapping_pd.to_csv('mapping.csv')
	
        # upload mapping
        if save_locally == False:
            print("upload mapping")
            with fs.open( bucket_destination + '/' + dataset_id + '/' + 'mapping.csv', 'wb' ) as f:
              mapping_pd.to_csv(f,index=False)

        if save_locally == False:
            print("upload spot images")
            fs.put(dataset_id + '_spot_images.zip',bucket_destination + '/' + dataset_id + '/version' + str(version) + '/spot_images.zip')
            os.remove(dataset_id + '_spot_images.zip')

        # save numpy
        a = 15
        images = ds_all.spot_images[:, :, 0, 30-a:30+a+1, 30-a:30+a+1].to_numpy()
        print(images.shape)
        np.save(dataset_id +'.npy',images)
            
        # remove intermidate result
        print("remove intermidiate files")
        shutil.rmtree(dir_in)
