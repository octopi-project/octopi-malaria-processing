import imageio
import cv2
import cupy as cp # conda install -c conda-forge cupy==10.2
import cupyx.scipy.ndimage
from skimage.feature.blob import _prune_blobs
import numpy as np
from scipy import signal

def imread_gcsfs(fs,file_path):
	img_bytes = fs.cat(file_path)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	return I

def resize_cp(ar,downsize_factor=4):
	# by Rinni
	s_ar = cp.zeros((int(ar.shape[0]/downsize_factor), int(ar.shape[0]/downsize_factor), 3))
	s_ar[:,:,0] = ar[:,:,0].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
	s_ar[:,:,1] = ar[:,:,1].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
	s_ar[:,:,2] = ar[:,:,2].reshape([int(ar.shape[0]/downsize_factor), int(downsize_factor), int(ar.shape[1]/downsize_factor), int(downsize_factor)]).mean(3).mean(1)
	return s_ar

def resize_image_cp(I,downsize_factor=4):
	I = I.astype('float')
	I_resized = cp.copy(I)
	I_resized = resize_cp(I_resized, downsize_factor)
	return(I_resized)

def remove_background(img_cpu, return_gpu_image=True):
	tophat = cv2.getStructuringElement(2, ksize=(17,17))
	tophat_gpu = cp.asarray(tophat)
	img_g_gpu = cp.asarray(img_cpu)
	img_th_gpu = img_g_gpu
	for k in range(3):
		img_th_gpu[:,:,k] = cupyx.scipy.ndimage.white_tophat(img_g_gpu[:,:,k], footprint=tophat_gpu)
	if return_gpu_image:
		return img_th_gpu
	else:
		return cp.asnumpy(img_th_gpu)

def gaussian_kernel_1d(n, std, normalized=True):
	if normalized:
		return cp.asarray(signal.gaussian(n, std))/(np.sqrt(2 * np.pi)*std)
	return cp.asarray(signal.gaussian(n, std))

def detect_spots(I, thresh = 12):
	# filters
	gauss_rs = np.array([4,6,8,10])
	gauss_sigmas = np.array([1,1.5,2,2.5])
	gauss_ts = np.divide(gauss_rs - 0.5,gauss_sigmas) # truncate value (to get desired radius)
	lapl_kernel = cp.array([[0,1,0],[1,-4,1],[0,1,0]])
	gauss_filters_1d = []
	for i in range(gauss_rs.shape[0]):
		gauss_filt_1d = gaussian_kernel_1d(gauss_rs[i]*2+1,gauss_sigmas[i],True)
		gauss_filt_1d = gauss_filt_1d.reshape(-1, 1)
		gauss_filters_1d.append(gauss_filt_1d)
	# apply all filters
	if len(I.shape) == 3:
		I = cp.average(I, axis=2, weights=cp.array([0.299,0.587,0.114]))
	filtered_imgs = []
	for i in range(len(gauss_filters_1d)): # apply LoG filters
		filt_img = cupyx.scipy.ndimage.convolve(I, gauss_filters_1d[i])
		filt_img = cupyx.scipy.ndimage.convolve(filt_img, gauss_filters_1d[i].transpose())
		filt_img = cupyx.scipy.ndimage.convolve(filt_img, lapl_kernel)
		filt_img *= -(gauss_sigmas[i]**2)
		filtered_imgs.append(filt_img)
	img_max_proj = cp.max(np.stack(filtered_imgs), axis=0)
	# return img_max_proj
	img_max_filt = cupyx.scipy.ndimage.maximum_filter(img_max_proj, size=3)
	# set pixels < thresh (12) to 0 (so they wont be in img_traceback)
	img_max_filt[img_max_filt < thresh] = 0 # check if uint8
	# origination masks
	img_traceback = cp.zeros(img_max_filt.shape)
	for i in range(len(filtered_imgs)): # trace back pixels to each filtered image
		img_traceback[img_max_filt == filtered_imgs[i]] = i+1
		img_traceback[img_max_filt == 0] = 0 # but make sure all pixels that were 0 are still 0
	ind = np.where(img_traceback != 0)
	spots = np.zeros((ind[0].shape[0],3)) # num spots x 3
	for i in range(ind[0].shape[0]):
		spots[i][0] = int(ind[1][i])
		spots[i][1] = int(ind[0][i])
		spots[i][2] = int(img_traceback[spots[i][1]][spots[i][0]])
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
