import argparse
import glob
import gcsfs
import os
import json
from utils import *

if __name__ == '__main__':

    image_format = 'png'

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
    parameters['a'] = 250 # size afte resizing

    debug_mode = True

    write_to_gcs = False

    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_source = 'gs://octopi-malaria-uganda-2022-data'
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
            parameters['row_end'] = 2
            parameters['column_end'] = 2

        Ny = parameters['row_end'] - parameters['row_start']
        Nx = parameters['column_end'] - parameters['column_start']
        a = parameters['a']

        # init. container
        I_overlay_WSI = np.zeros((Ny*parameters['a'],Nx*parameters['a'],3))
        I_left_WSI = np.zeros((Ny*parameters['a'],Nx*parameters['a']))
        I_right_WSI = np.zeros((Ny*parameters['a'],Nx*parameters['a']))
        I_low_NA_WSI = np.zeros((Ny*parameters['a'],Nx*parameters['a']))
        I_DPC_WSI = np.zeros((Ny*parameters['a'],Nx*parameters['a'],3))
        I_fluorescence_WSI = np.zeros((Ny*parameters['a'],Nx*parameters['a'],3))

        # dir_out = 'result/' + dataset_id
        # if not os.path.exists(dir_out):
        #     os.mkdir(dir_out)

        for i in range(parameters['row_start'],parameters['row_end']):
            for j in range(parameters['column_start'],parameters['column_end']):
                for k in range(parameters['z_start'],parameters['z_end']):
                    file_id = str(i) + '_' + str(j) + '_' + str(k)
                    print('processing fov ' + file_id)
                    
                    I_fluorescence = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'Fluorescence_405_nm_Ex.bmp')
                    I_fluorescence = cv2.cvtColor(I_fluorescence,cv2.COLOR_RGB2BGR)
                    I_fluorescence = I_fluorescence.astype('float')/255
                    I_fluorescence = I_fluorescence/flatfield_fluorescence
                    # I_fluorescence = I_fluorescence
                    I_fluorescence[I_fluorescence>1] = 1
                    I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
                    I_fluorescence = cv2.resize(I_fluorescence, (parameters['a'],parameters['a']), interpolation = cv2.INTER_AREA)
                    I_fluorescence_WSI[ 0+a*(Ny-1-i):a+a*(Ny-1-i) , 0+a*j:a+a*j , :] = I_fluorescence


                    I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
                    I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
                    # convert to mono if color
                    if len(I_BF_left.shape)==3:
                      I_BF_left = I_BF_left[:,:,1]
                      I_BF_right = I_BF_right[:,:,1]
                    I_BF_left = I_BF_left.astype('float')/255
                    I_BF_left = I_BF_left/flatfield_left
                    I_BF_right = I_BF_right.astype('float')/255
                    I_BF_right = I_BF_right/flatfield_right

                    I_BF_left = I_BF_left[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
                    I_BF_right = I_BF_right[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
                    I_BF_left = cv2.resize(I_BF_left, (parameters['a'],parameters['a']), interpolation = cv2.INTER_AREA)
                    I_BF_right = cv2.resize(I_BF_right, (parameters['a'],parameters['a']), interpolation = cv2.INTER_AREA)
                    I_left_WSI[ 0+a*(Ny-1-i):a+a*(Ny-1-i) , 0+a*j:a+a*j ] = I_BF_left
                    I_right_WSI[ 0+a*(Ny-1-i):a+a*(Ny-1-i) , 0+a*j:a+a*j ] = I_BF_right

                    # I_low_NA = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_low_NA.bmp')
                    # I_low_NA = I_low_NA.astype('float')/255
                    # I_low_NA = I_low_NA[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1']]
                    # I_low_NA = cv2.resize(I_low_NA, (parameters['a'],parameters['a']), interpolation = cv2.INTER_AREA)
                    # I_low_NA_WSI[ 0+a*(Ny-1-i):a+a*(Ny-1-i) , 0+a*j:a+a*j ] = I_low_NA

                    # generate dpc
                    I_DPC = generate_dpc(I_BF_left,I_BF_right)
                    if(len(I_DPC.shape)<3):
                        I_DPC = np.dstack((I_DPC,I_DPC,I_DPC))
                    I_DPC = cv2.resize(I_DPC, (parameters['a'],parameters['a']), interpolation = cv2.INTER_AREA)
                    I_DPC_WSI[ 0+a*(Ny-1-i):a+a*(Ny-1-i) , 0+a*j:a+a*j , :] = I_DPC

                    # overlay
                    I_overlay = 0.64*I_fluorescence + 0.36*I_DPC
                    I_overlay = cv2.resize(I_overlay, (parameters['a'],parameters['a']), interpolation = cv2.INTER_AREA)
                    I_overlay_WSI[ 0+a*(Ny-1-i):a+a*(Ny-1-i) , 0+a*j:a+a*j , :] = I_overlay

        cv2.imwrite(dir_out + '_fluorescence.' + image_format, I_fluorescence_WSI*255)
        cv2.imwrite(dir_out + '_left.' + image_format, I_left_WSI*255)
        cv2.imwrite(dir_out + '_right.' + image_format, I_right_WSI*255)
        # cv2.imwrite(dir_out + '/low_NA.' + image_format, I_low_NA_WSI*255)
        cv2.imwrite(dir_out + '_dpc.' + image_format, I_DPC_WSI*255)
        cv2.imwrite(dir_out + '_overlay.' + image_format, I_overlay_WSI*255)
