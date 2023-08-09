import argparse
import glob
import zarr
import gcsfs
import os
import json
import time
from utils import *

from cellpose import models
from cellpose import models, io

if __name__ == '__main__':

    image_format = 'png'

    parameters = {}
    parameters['crop_x0'] = 100
    parameters['crop_x1'] = 2900
    parameters['crop_y0'] = 100
    parameters['crop_y1'] = 2900
    parameters['a'] = 250 # size afte resizing

    debug_mode = False

    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_source = 'gs://octopi-malaria-tanzania-2021-data' + '/Negative-Donor-Samples'
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

        # print(dataset_id)

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

        # dataframe for number of cells per FOV
        segmantation_stat_pd = pd.DataFrame(columns=['FOV_row','FOV_col','count'])
        total_number_of_cells = 0

        # cellpose
        cellpose_model_path = 'cp_dpc_new'
        model = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)

        t0 = time.time()

        for i in range(parameters['row_start'],parameters['row_end']):
            for j in range(parameters['column_start'],parameters['column_end']):
                for k in range(parameters['z_start'],parameters['z_end']):

                    file_id = str(i) + '_' + str(j) + '_' + str(k)
                    # print('processing fov ' + file_id)

                    # generate dpc
                    I_BF_left = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp')
                    I_BF_right = imread_gcsfs(fs,bucket_source + '/' + dataset_id + '/0/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp')
                    if len(I_BF_left.shape)==3: # convert to mono if color
                      I_BF_left = I_BF_left[:,:,1]
                      I_BF_right = I_BF_right[:,:,1]
                    I_BF_left = I_BF_left.astype('float')/255
                    I_BF_right = I_BF_right.astype('float')/255
                    I_DPC = generate_dpc(I_BF_left,I_BF_right)
                    
                    # if(len(I_DPC.shape)<3):
                    #     I_DPC = np.dstack((I_DPC,I_DPC,I_DPC))

                    # segmentation
                    im = I_DPC - np.min(I_DPC)
                    im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
                    mask, flows, styles = model.eval(im, diameter=None)

                    # count the number of cells
                    number_of_cells = np.amax(mask)
                    FOV_entry = pd.DataFrame.from_dict({'FOV_row':[i],'FOV_col':[j],'count':[number_of_cells]})
                    segmantation_stat_pd = pd.concat([segmantation_stat_pd,FOV_entry])
                    total_number_of_cells = total_number_of_cells + number_of_cells
                      
                    # save mask
                    mask = mask > 0
                    mask_uint8 = mask.astype('uint8')*255
                    with fs.open( bucket_destination + '/' + dataset_id + '/' + 'segmentation_mask_binary/' + file_id + '.bmp', 'wb' ) as f:
                        f.write(cv2.imencode('.bmp',mask_uint8)[1].tobytes())
                    # cv2.imwrite('mask.png',mask_uint8)

        with fs.open( bucket_destination + '/' + dataset_id + '/segmentation_stat.csv', 'wb') as f:
            segmantation_stat_pd.to_csv(f,index=False)
        with fs.open( bucket_destination + '/' + dataset_id + '/total number of RBCs.txt', 'w') as f:
            f.write(str(total_number_of_cells))
        print( dataset_id + ': ' + str(total_number_of_cells) )
    print(time.time()-t0)