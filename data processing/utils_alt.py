import imageio
import cv2
import numpy as np
import scipy
import scipy.ndimage as ndimage
from scipy.ndimage import laplace
from skimage.feature.blob import _prune_blobs
from scipy import signal
import pandas as pd

def imread_gcsfs(fs,file_path):
	img_bytes = fs.cat(file_path)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	return I

def resize_cp(ar,downsize_factor=4):
	# by Rinni
	s_ar = np.zeros((int(ar.shape[0]/downsize_factor), int(ar.shape[0]/downsize_factor), 3))
	s_ar[:,:,0] = ar[:,:,0].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
	s_ar[:,:,1] = ar[:,:,1].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
	s_ar[:,:,2] = ar[:,:,2].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
	return s_ar

def resize_image_cp(I,downsize_factor=4):
	I = I.astype('float')
	I_resized = np.copy(I)
	I_resized = resize_cp(I_resized, downsize_factor)
	return(I_resized)

def remove_background(img, return_gpu_image=True):
	tophat = cv2.getStructuringElement(2, ksize=(17,17))
	tophat_gpu = np.asarray(tophat)
	img_th = img.copy()
	for k in range(3):
		img_th[:,:,k] = scipy.ndimage.white_tophat(img[:,:,k], footprint=tophat_gpu)
	return img_th

def gaussian_kernel_1d(n, std, normalized=True):
	if normalized:
		return signal.gaussian(n, std)/(np.sqrt(2 * np.pi)*std)
	return signal.gaussian(n, std)

def detect_spots(I, thresh = 12):
	# filters
	gauss_rs = np.array([4,6,8,10])
	gauss_sigmas = np.array([1,1.5,2,2.5])
	gauss_ts = np.divide(gauss_rs - 0.5,gauss_sigmas) # truncate value (to get desired radius)
	lapl_kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]])
	gauss_filters_1d = []
	for i in range(gauss_rs.shape[0]):
		gauss_filt_1d = gaussian_kernel_1d(gauss_rs[i]*2+1,gauss_sigmas[i],True)
		gauss_filt_1d = gauss_filt_1d.reshape(-1, 1)
		gauss_filters_1d.append(gauss_filt_1d)
	# apply all filters
	if len(I.shape) == 3:
		I = np.average(I, axis=2, weights=np.array([0.299,0.587,0.114]))
	filtered_imgs = []
	for i in range(len(gauss_filters_1d)): # apply LoG filters
		filt_img = scipy.ndimage.convolve(I, gauss_filters_1d[i])
		filt_img = scipy.ndimage.convolve(filt_img, gauss_filters_1d[i].transpose())
		filt_img = scipy.ndimage.convolve(filt_img, lapl_kernel)
		filt_img *= -(gauss_sigmas[i]**2)
		filtered_imgs.append(filt_img)
	img_max_proj = np.max(np.stack(filtered_imgs), axis=0)
	# return img_max_proj
	img_max_filt = scipy.ndimage.maximum_filter(img_max_proj, size=3)
	# set pixels < thresh (12) to 0 (so they wont be in img_traceback)
	img_max_filt[img_max_filt < thresh] = 0 # check if uint8
	# origination masks
	img_traceback = np.zeros(img_max_filt.shape)
	for i in range(len(filtered_imgs)): # trace back pixels to each filtered image
		img_traceback[img_max_filt == filtered_imgs[i]] = i+1
		img_traceback[img_max_filt == 0] = 0 # but make sure all pixels that were 0 are still 0
	ind = np.where(img_traceback != 0)
	spots = np.zeros((ind[0].shape[0],3)) # num spots x 3
	for i in range(ind[0].shape[0]):
		spots[i][0] = int(ind[1][i])
		spots[i][1] = ind[0][i]
		spots[i][2] = int(img_traceback[int(spots[i][1])][int(spots[i][0])])
	spots = spots.astype(int)
	return spots

# filter spots to avoid overlapping ones
def prune_blobs(spots_list):
	overlap = .5
	num_sigma = 4
	min_sigma = 1
	max_sigma = 2.5
	scale = np.linspace(0, 1, num_sigma)[:, np.newaxis]
	sigma_list = scale * (max_sigma - min_sigma) + min_sigma
	# translate final column of lm, which contains the index of the
	# sigma that produced the maximum intensity value, into the sigma
	sigmas_of_peaks = sigma_list[spots_list[:, -1]-1]
	# select one sigma column, keeping dimension
	sigmas_of_peaks = sigmas_of_peaks[:, 0:1]
	# Remove sigma index and replace with sigmas
	spots_list = np.hstack([spots_list[:,:-1], sigmas_of_peaks])
	result_pruned = _prune_blobs(spots_list, overlap)
	return result_pruned

def highlight_spots(I,spot_list,contrast_boost=1.6):
	# bgremoved_fluorescence_spotBoxed = np.copy(bgremoved_fluorescence)
	I = I.astype('float')/255 # this copies the image
	I = I*contrast_boost # enhance contrast
	for s in spot_list:
		add_bounding_box(I,int(s[0]),int(s[1]),int(s[2]))
	return I

def add_bounding_box(I,x,y,r,extension=2,color=[0.6,0.6,0]):
	ny, nx, nc = I.shape
	x_min = max(x - r - extension,0)
	y_min = max(y - r - extension,0)
	x_max = min(x + r + extension,nx-1)
	y_max = min(y + r + extension,ny-1)
	for i in range(3):
		I[y_min,x_min:x_max+1,i] = color[i]
		I[y_max,x_min:x_max+1,i] = color[i]
		I[y_min:y_max+1,x_min,i] = color[i]
		I[y_min:y_max+1,x_max,i] = color[i]

def remove_spots_in_masked_regions(spotList,mask):
	mask = mask.astype('float')/255
	mask = np.sum(mask,axis=-1) # masked out region has pixel value 0 ;# mask[mask>0] = 1 #         cv2.imshow('mask',mask) # cv2.waitKey(0)
	for s in spotList:
		x = s[0]
		y = s[1]
		if mask[int(y),int(x)] == 0:
			s[-1] = 0
	spot_list = np.array([s for s in spotList if s[-1] > 0])
	return spot_list

def extract_spot_data(I_background_removed,I_raw,spot_list,i,j,k,settings,extension=1):
	downsize_factor=settings['spot_detection_downsize_factor']
	extension = extension*downsize_factor
	ny, nx, nc = I_background_removed.shape
	I_background_removed = I_background_removed.astype('float')
	I_raw = I_raw/255
	columns = ['FOV_row','FOV_col','FOV_z','x','y','r','R','G','B','R_max','G_max','B_max','lap_total','lap_max','numPixels','numSaturatedPixels','idx']
	spot_data_pd = pd.DataFrame(columns=columns)
	idx = 0
	for s in spot_list:
		# get spot
		x = int(s[0])
		y = int(s[1])
		r = s[2]
		x_min = max(int((x - r - extension)),0)
		y_min = max(int((y - r - extension)),0)
		x_max = min(int((x + r + extension)),nx-1)
		y_max = min(int((y + r + extension)),ny-1)
		cropped = I_background_removed[y_min:(y_max+1),x_min:(x_max+1),:]
		cropped_raw = I_raw[y_min:(y_max+1),x_min:(x_max+1),:]
		# extract spot data
		B = np.sum(cropped[:,:,2])
		G = np.sum(cropped[:,:,1])
		R = np.sum(cropped[:,:,0])
		B_max = np.max(cropped[:,:,2])
		G_max = np.max(cropped[:,:,1])
		R_max = np.max(cropped[:,:,0])
		lap = laplace(np.sum(cropped,2))
		lap_total = np.sum(np.abs(lap))
		lap_max = np.max(np.abs(lap))
		numPixels = cropped[:,:,0].size
		numSaturatedPixels = np.sum(cropped_raw == 1)
		# add spot entry
		spot_entry = pd.DataFrame.from_dict({'FOV_row':[i],'FOV_col':[j],'FOV_z':[k],'x':[x],'y':[y],'r':[r],'R':[R],'G':[G],'B':[B],'R_max':[R_max],'G_max':[G_max],'B_max':[B_max],'lap_total':[lap_total],'lap_max':[lap_max],'numPixels':[numPixels],'numSaturatedPixels':[numSaturatedPixels],'idx':[idx]})
		# spot_data_pd = spot_data_pd.append(spot_entry, ignore_index=True, sort=False)
		spot_data_pd = pd.concat([spot_data_pd,spot_entry])
		# increment idx
		idx = idx + 1
	return spot_data_pd

def process_spots(I_background_removed,I_raw,spot_list,i,j,k,settings,I_mask=None):
	# get rid of spots in masked out regions
	if I_mask!=None:
		spot_list = remove_spots_in_masked_regions(spot_list,I_mask)
	# extract spot statistics
	spot_data_pd = extract_spot_data(I_background_removed,I_raw,spot_list,i,j,k,settings)
	return spot_list, spot_data_pd

