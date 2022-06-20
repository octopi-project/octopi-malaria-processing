import cv2
import numpy as np
import pyvips # https://www.libvips.org/install.html
import gcsfs
import argparse
import json
import os
from utils import *

# load flatfield
flatfield_fluorescence = np.load('illumination correction/flatfield_fluorescence.npy')
flatfield_fluorescence = np.dstack((flatfield_fluorescence,flatfield_fluorescence,flatfield_fluorescence))
flatfield_left = np.load('illumination correction/flatfield_left.npy')
flatfield_right = np.load('illumination correction/flatfield_right.npy')

parameters = {}
parameters['crop_x0'] = 100
parameters['crop_x1'] = 2900
parameters['crop_y0'] = 100
parameters['crop_y1'] = 2900

gcs_project = 'soe-octopi'
gcs_token = 'data-20220317-keys.json'
gcs_settings = {}
gcs_settings['gcs_project'] = gcs_project
gcs_settings['gcs_token'] = gcs_token

debug_mode = True

gcs_project = 'soe-octopi'
gcs_token = 'data-20220317-keys.json'
gcs_settings = {}
gcs_settings['gcs_project'] = gcs_project
gcs_settings['gcs_token'] = gcs_token

bucket_source = 'gs://octopi-malaria-tanzania-2021-data'


if __name__ == '__main__':

    fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    # load the list of dataset
    if args.data_id != None:
        DATASET_ID = [args.data_id]
    else:
        f = open('list of datasets.txt','r')
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
        f.close()

    # go through dataset
    for dataset_id in DATASET_ID:

        print(dataset_id)
        json_file = fs.cat(bucket_source + '/' + dataset_id + '/acquisition parameters.json')
        acquisition_parameters = json.loads(json_file)

        parameters['row_start'] = 0
        parameters['row_end'] = acquisition_parameters['Ny']
        parameters['column_start'] = 0
        parameters['column_end'] = acquisition_parameters['Nx']
        parameters['z_start'] = 0
        parameters['z_end'] = acquisition_parameters['Nz']
        if debug_mode:
            parameters['row_end'] = 2
            parameters['column_end'] = 2

        w = parameters['column_end'] - parameters['column_start']
        h = parameters['row_end'] - parameters['row_start']

        # output dir
        dir_out = '/Users/hongquanli/Downloads/' + dataset_id
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

        # vimgs
        vimgs_fluorescence = []
        vimgs_DPC = []
        vimgs_overlay = []

        # go through the scan
        for i in range(parameters['row_end']-1,parameters['row_start']-1,-1):
            for j in range(parameters['column_start'],parameters['column_end']):
                for k in range(parameters['z_start'],parameters['z_end']):
                    file_id = str(i) + '_' + str(j) + '_' + str(k)
                    print('processing fov ' + file_id)

                    I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
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

                    # enhance contrast
                    I_fluorescence = I_fluorescence*1.4
                    I_fluorescence[I_fluorescence>1] = 1

                    # illumination correction
                    I_fluorescence = I_fluorescence/flatfield_fluorescence
                    I_BF_left = I_BF_left/flatfield_left
                    I_BF_right = I_BF_right/flatfield_right
                    
                    # crop image
                    I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
                    I_BF_left = I_BF_left[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
                    I_BF_right = I_BF_right[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
                    
                    # generate dpc
                    I_DPC = generate_dpc(I_BF_left,I_BF_right)
                    if(len(I_DPC.shape)<3):
                        I_DPC = np.dstack((I_DPC,I_DPC,I_DPC))
                    
                    # overlay
                    I_fluorescence = I_fluorescence*255
                    I_DPC = I_DPC*255
                    I_overlay = (0.64*I_fluorescence + 0.36*I_DPC)

                    # vips
                    tmp_overlay_v = pyvips.Image.new_temp_file(str(i)+'_'+str(j)+'.v')
                    tmp = pyvips.Image.new_from_array(I_overlay.astype('uint8'))
                    tmp.write(tmp_overlay_v)
                    vimgs_overlay.append(tmp_overlay_v)

                    tmp_DPC_v = pyvips.Image.new_temp_file(str(i)+'_'+str(j)+'.v')
                    tmp = pyvips.Image.new_from_array(I_DPC.astype('uint8'))
                    tmp.write(tmp_DPC_v)
                    vimgs_DPC.append(tmp_DPC_v)

                    tmp_fluorescence_v = pyvips.Image.new_temp_file(str(i)+'_'+str(j)+'.v')
                    tmp = pyvips.Image.new_from_array(I_fluorescence.astype('uint8'))
                    tmp.write(tmp_fluorescence_v)
                    vimgs_fluorescence.append(tmp_fluorescence_v)

        print('joining arrays')
        vimgs_fluorescence = pyvips.Image.arrayjoin(vimgs_fluorescence, across=w)
        vimgs_DPC = pyvips.Image.arrayjoin(vimgs_DPC, across=w)
        vimgs_overlay = pyvips.Image.arrayjoin(vimgs_overlay, across=w)

        print('writing to files')
        # vimg.write_to_file( dataset_id + '.tiff', tile=True, tile_width=1024, tile_height=1024,pyramid=True,compression='none')
        # https://www.libvips.org/API/current/VipsForeignSave.html#vips-dzsave
        dataset_id = ''.join(dataset_id.split('.')[:-1]) # get rid of fractional seconds
        vimgs_fluorescence.dzsave(dataset_id+'_fluorescence',tile_size=1024,suffix='.jpg[Q=95]')
        vimgs_DPC.dzsave(dataset_id+'_dpc',tile_size=1024,suffix='.jpg[Q=95]')
        vimgs_overlay.dzsave(dataset_id+'_overlay',tile_size=1024,suffix='.jpg[Q=95]')
