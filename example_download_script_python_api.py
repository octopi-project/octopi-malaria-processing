import os
from google.cloud import storage
import gcsfs
import cv2
import imageio
import time

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "processing-20220215-keys.json"

# pip3 install gcsfs google-cloud-storage opencv-python

def list_subdirectories(bucket, directory):
	if(directory == './'):
		directory = ''
	iterator = bucket.list_blobs(delimiter='/', prefix=directory)
	response = iterator._get_next_page_response()
	return response['prefixes']

def download_blob(bucket, source_blob_name, destination_file_name):
	blob = bucket.blob(source_blob_name)
	blob.download_to_filename(destination_file_name)
	print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))

storage_client = storage.Client()
bucket = storage_client.bucket('octopi-malaria-tanzania-2021-data')
'''
subdirs = list_subdirectories(bucket,'./')
for subdir in subdirs:
    print(subdir)
print('---')
dataset = 'U3D_201910_2022-01-11_23-11-36.799392'
items = list_subdirectories(bucket,dataset)
for items in subdirs:
    print(items)
'''

source_blob_name = 'U3D_201910_2022-01-11_23-11-36.799392/0/9_12_0_BF_LED_matrix_low_NA.bmp'

time_elapsed = 0
N = 25
for i in range(N):
	t0 = time.time()
	download_blob(bucket,source_blob_name,'tmp.bmp')
	I = cv2.imread('tmp.bmp')
	t1 = time.time()
	print(t1-t0)
	time_elapsed = time_elapsed + (t1-t0)
print('average read time is ' + str(time_elapsed/N))
