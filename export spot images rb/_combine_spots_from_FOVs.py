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

    bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
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

        dir_in = 'spot images_' + dataset_id
        files = glob.glob(dir_in + '/' + '*.zarr')
        counter = 0
        for file in files:
            print(file.split('/')[-1])
            ds = xr.open_zarr(file)
            if counter == 0:
                data_all = ds.spot_images
            else:
                data_all = xr.concat([data_all,ds.spot_images],dim='t')
            counter = counter + 1
        ds_all = xr.Dataset({'spot_images':data_all})
        if write_to_gcs:
            if use_zip_store:
                with zarr.ZipStore(dataset_id + '_spot_images.zip', mode='w') as store:
                    ds_all.to_zarr(store, mode='w')
                fs.put(dataset_id + '_spot_images.zip',bucket_destination + '/' + dataset_id + '/' + 'spot_images.zip')
                os.remove(dataset_id + '_spot_images.zip')
                '''
                # direct zip store - this does not work: TypeError: expected str, bytes or os.PathLike object, not GCSFile
                with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_images.zip', 'wb' ) as f:
                    with zarr.ZipStore(f, mode='w') as store:
                        ds_all.to_zarr(store, mode='w')
                '''
            else:
                # directory store
                store = fs.get_mapper(bucket_destination + '/' + dataset_id + '/' + 'spot_images.zarr')
                ds_all.to_zarr(store, mode='w')
        else:
            if use_zip_store:
                # zip store
                with zarr.ZipStore(dataset_id + '_spot_images.zip', mode='w') as store:
                    ds_all.to_zarr(store, mode='w')
            else:
                # directory store
                ds_all.to_zarr(dataset_id + '_spot_images.zarr', mode='w')