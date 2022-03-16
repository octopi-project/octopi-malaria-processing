import os
import gcsfs
import cv2
import imageio
import time
import json

# gcsfs
print('------')
fs = gcsfs.GCSFileSystem(project='soe-octopi',token='processing-20220215-keys.json')

# access individual files
filename = 'gs://octopi-malaria-tanzania-2021-data/U3D_201910_2022-01-11_23-11-36.799392/acquisition parameters.json'
json_file = fs.cat(filename)
acquisition_parameters = json.loads(json_file)
print(acquisition_parameters)