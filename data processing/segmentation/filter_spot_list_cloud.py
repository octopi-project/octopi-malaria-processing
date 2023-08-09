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
import FlowCal
import matplotlib.pyplot as plt
import cv2

def imread_gcsfs(fs,file_path):
    img_bytes = fs.cat(file_path)
    I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
    return I

version = 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    export_selected_spots = False # selected spots vs all spots
    save_intermediate_to_gcs = False # spot saved to the local folder: spot images_DATASET ID
    combine_zarr = True

    debug_mode = True
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

    bucket_source = 'gs://octopi-malaria-tanzania-2021-data/Negative-Donor-Samples'
    bucket_destination = 'gs://octopi-malaria-data-processing'

    settings = {}
    settings['export selected spots'] = export_selected_spots
    settings['bucket_source'] = bucket_source
    settings['bucket_destination'] = bucket_destination
    settings['save to gcs'] = save_intermediate_to_gcs

    crop_offset_x = 100
    crop_offset_y = 100

    dir_segmentation_mask = '/Volumes/T7/malaria-tanzina-2021/dpc/Negative-Donor-Samples'

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
            parameters['row_end'] = 10
            parameters['column_end'] = 10
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

        # add a new column
        spot_data_pd['Overlap'] = np.NaN

        # # go through the FOVs
        # for i in range(parameters['row_start'],parameters['row_end']):
        #     for j in range(parameters['column_start'],parameters['column_end']):

        #         # get the dataframe 
        #         idx = (spot_data_pd['FOV_row']==i) & (spot_data_pd['FOV_col']==j)
        #         spot_data_current_fov = spot_data_pd[idx]

        i_current = -1
        j_current = -1

        # select FOVs
        idx = (spot_data_pd['FOV_row'] >= parameters['row_start']) & (spot_data_pd['FOV_row'] < parameters['row_end']) & (spot_data_pd['FOV_col'] >= parameters['column_start']) & (spot_data_pd['FOV_col'] < parameters['column_end'])
        spot_data_pd = spot_data_pd[idx]

        for k, row in spot_data_pd.iterrows():

            # get the FOV
            i = int(row['FOV_row'])
            j = int(row['FOV_col'])
            x = int(row['x'])
            y = int(row['y'])

            # load the segmentation mask
            if i != i_current or j != j_current:
                '''
                # for debugging
                try:
                    cv2.imwrite('mask.png',mask)
                except:
                    pass
                '''
                # print(str(i) + ' ' + str(j))
                
                # mask = cv2.imread(dir_segmentation_mask + '/segmentation/' + dataset_id + '/0/' + str(i) + '_' + str(j) + '_mask.bmp')
                mask = imread_gcsfs(fs,'gs://octopi-malaria-data-processing/malaria-tanzina-2021/Negative-Donor-Samples/' + dataset_id + '/0/' + file_id + '_f_BF_LED_matrix_dpc_mask.bmp')
                mask = utils.gcs
                mask = mask[:,:,0]
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.erode(mask, kernel) # erode
                # print(dir_segmentation_mask + '/segmentation/' + dataset_id + '/0/' + str(i) + '_' + str(j) + '_mask.bmp')
                i_current = i
                j_current = j

            '''
            # for debugging - mark the spot in the mask for debugging
            mask[100+y-1:100+y+2,100+x-1:100+x+2,1] = 255
            mask[100+y-1:100+y+2,100+x-1:100+x+2,0] = 0
            mask[100+y-1:100+y+2,100+x-1:100+x+2,2] = 0
            '''

            # go through the spot to check overlap
            spot_data_pd.at[k,'Overlap'] = np.sum( mask[crop_offset_y+y-1:crop_offset_y+y+2,crop_offset_x+x-1:crop_offset_x+x+2]>0 ) / 9

        # save updated spot list
        spot_data_pd.to_csv(dataset_id + '.csv')
        n = spot_data_pd.shape[0]
        spot_data_pd = spot_data_pd[spot_data_pd['Overlap']>=6/9]
        n1 = spot_data_pd.shape[0]
        spot_data_pd.to_csv(dataset_id + '_in_cell.csv')
        print(str(n1) + '/' + str(n))
