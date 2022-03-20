import os
import gcsfs
import pandas as pd

gcs_project = 'soe-octopi'
gcs_token = 'data-20220317-keys.json'
gcs_settings = {}
gcs_settings['gcs_project'] = gcs_project
gcs_settings['gcs_token'] = gcs_token

# dataset ID
bucket_source = 'gs://octopi-malaria-tanzania-2021-data'
bucket_destination = 'gs://octopi-malaria-data-processing'
dataset_id = 'U3D_201910_2022-01-11_23-11-36.799392'

fs = gcsfs.GCSFileSystem(project=gcs_settings['gcs_project'],token=gcs_settings['gcs_token'])

file_id = '0_0_0'

with fs.open( bucket_destination + '/' + dataset_id + '/' + 'spot_data/' + file_id + '.csv', 'r' ) as f:
    spotData_all_pd = pd.read_csv(f, index_col=None, header=0)
print(spotData_all_pd)

