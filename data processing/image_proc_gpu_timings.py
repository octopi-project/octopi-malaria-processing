import os
import cv2
import time
import imageio
from utils import *
import matplotlib.pyplot as plt

fov_path = '0_0_0_fl.bmp'
settings = {}
settings['spot_detection_downsize_factor'] = 4
settings['saving_file_format'] = 'bmp'
settings['saving_location'] = 'local'
settings['downsize_factor'] = 4
settings['spot_detection_threshold'] = 10
settings['save_spot_detection_visualization'] = True

# regions to process + other settings
parameters = {}
parameters['crop_x0'] = 100
parameters['crop_x1'] = 2900
parameters['crop_y0'] = 100
parameters['crop_y1'] = 2900

# pull in image
I_fluorescence = imageio.imread(fov_path)

# crop image
t00 = time.time()
I_fluorescence = I_fluorescence[ parameters['crop_y0']:parameters['crop_y1'], parameters['crop_x0']:parameters['crop_x1'], : ]
t1 = time.time()
print('cropping took ' + str(t1-t00) + 's')

# remove background
t0 = time.time()
I_fluorescence_bg_removed = remove_background(I_fluorescence,return_gpu_image=True)
t1 = time.time()
print('removing background took ' + str(t1-t0) + 's')

# detect spots
t0 = time.time()
spot_list = detect_spots(resize_image_cp(I_fluorescence_bg_removed,downsize_factor=settings['spot_detection_downsize_factor']),thresh=settings['spot_detection_threshold'])
if(len(spot_list)==0):
    print('no spots!')
spot_list = prune_blobs(spot_list)
t1 = time.time()
print('detecting spots took ' + str(t1-t0) + 's')

# scale coordinates for full-res image
spot_list = spot_list*settings['spot_detection_downsize_factor']

# process spots
t0 = time.time()
spot_list, spot_data_pd = process_spots(I_fluorescence_bg_removed,I_fluorescence,spot_list,0,0,0,settings)
t1 = time.time()
print('processing spots took ' + str(t1-t0) + 's')

# get boxed image
t0 = time.time()
I_boxed = cv2.cvtColor(cp.asnumpy(255*highlight_spots(I_fluorescence_bg_removed,spot_list)).astype('uint8'),cv2.COLOR_RGB2BGR)
t11 = time.time()
print('boxing spots took ' + str(t11-t0) + 's')

print('everything took ' + str(t11-t00) + 's')
print(spot_data_pd)
print(spot_list)
plt.imshow(I_boxed)