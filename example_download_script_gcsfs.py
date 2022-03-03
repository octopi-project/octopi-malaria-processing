import os
import gcsfs
import cv2
import imageio
import time

# gcsfs
print('------')
fs = gcsfs.GCSFileSystem(project='soe-octopi',token='processing-20220215-keys.json')

# list datasets
for folder in fs.ls('octopi-malaria-tanzania-2021-data'):
	print(folder)
print('------')

# access individual files
filename = 'octopi-malaria-tanzania-2021-data/U3D_201910_2022-01-11_23-11-36.799392/0/9_12_0_BF_LED_matrix_low_NA.bmp'
print(fs.ls(filename))

time_elapsed = 0
N = 5
for i in range(N):
	t0 = time.time()
	img_bytes = fs.cat(filename)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	t1 = time.time()
	print(t1-t0)
	time_elapsed = time_elapsed + (t1-t0)
print('average read time is ' + str(time_elapsed/N))


# cv2.imshow('',I)
# cv2.waitKey(0)



# multiprocessing: https://stackoverflow.com/questions/66283634/use-gcsfilesystem-with-multiprocessing

# # file_path = str(data_dir / f"{file_name}{ch}")
# file_path = 'MKZ210008_DAPI_20x_26x26_2022-01-08_22-42-42.166106/0/0_0_0BF LED matrix full.bmp'
# url = await octopi_image_storage.generate_presigned_url(file_path)
# fimage = imageio.imread(url)
