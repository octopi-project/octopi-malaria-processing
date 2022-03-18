import imageio
import cv2
import cupy as cp # conda install -c conda-forge cupy==10.2
import cupyx.scipy.ndimage

def imread_gcsfs(fs,file_path):
	img_bytes = fs.cat(file_path)
	I = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
	return I

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

