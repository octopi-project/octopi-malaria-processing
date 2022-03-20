import imageio
import cv2
# import cupy as cp # conda install -c conda-forge cupy==10.2
# import cupyx.scipy.ndimage
import numpy as np
from scipy import signal
import pandas as pd

def imread_gcsfs(fs,file_path):
	img_bytes = fs.cat(file_path)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	return I

def generate_dpc(I1,I2,use_gpu=False):
	if use_gpu:
		# img_dpc = cp.divide(img_left_gpu - img_right_gpu, img_left_gpu + img_right_gpu)
		# to add
		I_dpc = 0
	else:
		I_dpc = np.divide(I1-I2,I1+I2)
		I_dpc = I_dpc + 0.5
	return I_dpc

def process_fov_for_spot_visualizations(I_fluorescence,I_dpc,spot_data,settings,parameters,dir_out,r=30,scale=5,image_format='jpeg'):
	# make I_dpc RGB
	if(len(I_dpc.shape)<3):
		I_dpc = np.dstack((I_dpc,I_dpc,I_dpc))
	# get overlay
	I_overlay = 0.64*I_fluorescence + 0.36*I_dpc
	# get the full image size
	height,width,channels = I_fluorescence.shape
	# go through spot
	for idx, entry in spot_data.iterrows():
		# get coordinate
		i = int(entry['FOV_row'])
		j = int(entry['FOV_col'])
		x = int(entry['x'])*settings['spot_detection_downsize_factor'] # to change upstream
		y = int(entry['y'])*settings['spot_detection_downsize_factor'] # to change upstream
		# create the arrays for cropped images
		I_DPC_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_fluorescence_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		I_overlay_cropped = np.zeros((2*r+1,2*r+1,3), np.float)
		# identify cropping region in the full FOV 
		x_start = max(0,x-r)
		x_end = min(x+r,width-1)
		y_start = max(0,y-r)
		y_end = min(y+r,height-1)
		x_idx_FOV = slice(x_start,x_end+1)
		y_idx_FOV = slice(y_start,y_end+1)
		# identify cropping region in the cropped images
		x_cropped_start = x_start - (x-r)
		x_cropped_end = (2*r+1-1) - ((x+r)-x_end)
		y_cropped_start = y_start - (y-r)
		y_cropped_end = (2*r+1-1) - ((y+r)-y_end)
		x_idx_cropped = slice(x_cropped_start,x_cropped_end+1)
		y_idx_cropped = slice(y_cropped_start,y_cropped_end+1)
		# do the cropping 
		I_DPC_cropped[y_idx_cropped,x_idx_cropped,:] = I_dpc[y_idx_FOV,x_idx_FOV,:]
		I_fluorescence_cropped[y_idx_cropped,x_idx_cropped,:] = I_fluorescence[y_idx_FOV,x_idx_FOV,:]
		I_overlay_cropped[y_idx_cropped,x_idx_cropped,:] = I_overlay[y_idx_FOV,x_idx_FOV,:]
		# put the images in a row
		row = np.hstack((I_DPC_cropped,I_fluorescence_cropped,I_overlay_cropped))
		row = cv2.resize(row,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
		cv2.imwrite(dir_out + '/' + str(i) + '_' + str(j) + '_' + str(x) + '_' + str(y) + '.' + image_format, row*255)