import xarray as xr
import argparse
import glob
import zarr
import gcsfs
import os

if __name__ == '__main__':

    write_to_gcs = True
    use_zip_store = True

    gcs_project = 'soe-octopi'
    gcs_token = 'data-20220317-keys.json'
    gcs_settings = {}
    gcs_settings['gcs_project'] = gcs_project
    gcs_settings['gcs_token'] = gcs_token

    bucket_destination = 'gs://octopi-malaria-data-processing'

    if write_to_gcs or read_from_gcs:
        fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])

    parser = argparse.ArgumentParser()
    parser.add_argument("data_id",nargs='?',help="input data id")
    args = parser.parse_args()

    if args.data_id != None:
        DATASET_ID = [args.data_id]
    else:
        f = open('list of datasets_negative.txt','r')
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
        f.close()

    for dataset_id in DATASET_ID:
        print('downloading spot images for ' + dataset_id)
        fs.get(bucket_destination + '/' + dataset_id + '/' + 'spot_images.zip',dataset_id + '_spot_images.zip')
