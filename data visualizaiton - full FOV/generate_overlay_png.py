import argparse
import glob
import zarr
import gcsfs
import os
import json
from utils import *

correct_illumination = True
if correct_illumination:
	# load flatfield
	flatfield_fluorescence = np.load('illumination correction/flatfield_fluorescence.npy')
	flatfield_fluorescence = np.dstack((flatfield_fluorescence,flatfield_fluorescence,flatfield_fluorescence))
	flatfield_left = np.load('illumination correction/flatfield_left.npy')
	flatfield_right = np.load('illumination correction/flatfield_right.npy')

if __name__ == '__main__':

    image_format = 'png'

    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900

    debug_mode = True

    write_to_gcs = False
    use_zip_store = True

    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
    bucket_destination = 'gs://octopi-malaria-data-processing'

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
            parameters['row_end'] = 5
            parameters['column_end'] = 5

        dir_out = './' + dataset_id
        if not os.path.exists(dir_out):
            os.mkdir(dir_out)

        for i in range(parameters['row_start'],parameters['row_end']):
            for j in range(parameters['column_start'],parameters['column_end']):
                for k in range(parameters['z_start'],parameters['z_end']):
                    file_id = str(i) + '_' + str(j) + '_' + str(k)
                    print('processing fov ' + file_id)

                    I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
                    I_fluorescence = cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR)
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
                    if correct_illumination:
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
                    I_overlay = 0.64*I_fluorescence + 0.36*I_DPC

                    cv2.imwrite(dir_out + '/' + str(i) + '_' + str(j) + '.' + image_format, I_overlay*255)
        
        if write_to_gcs:
            if use_zip_store:
                '''
                with zarr.ZipStore(dataset_id + '_spot_images.zip', mode='w') as store:
                    ds_all.to_zarr(store, mode='w')
                fs.put(dataset_id + '_spot_images.zip',bucket_destination + '/' + dataset_id + '/' + 'spot_images.zip')
                os.remove(dataset_id + '_spot_images.zip')
                '''
                pass
            else:
                # directory store
                pass
                '''
                store = fs.get_mapper(bucket_destination + '/' + dataset_id + '/' + 'spot_images.zarr')
                ds_all.to_zarr(store, mode='w')
                '''
        else:
            if use_zip_store:
                '''
                # zip store
                with zarr.ZipStore(dataset_id + '_spot_images.zip', mode='w') as store:
                    ds_all.to_zarr(store, mode='w')
                '''
                pass
            else:
                # directory store
                # ds_all.to_zarr(dataset_id + '_spot_images.zarr', mode='w')
                pass
