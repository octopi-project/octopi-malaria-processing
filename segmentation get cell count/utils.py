import imageio
import numpy as np

def generate_dpc(im_left, im_right):
    # Normalize the images
    im_left = im_left.astype(float)/255
    im_right = im_right.astype(float)/255
    # differential phase contrast calculation
    im_dpc = 0.5 + np.divide(im_left-im_right, im_left+im_right)
    # take care of errors
    im_dpc[im_dpc < 0] = 0
    im_dpc[im_dpc > 1] = 1
    im_dpc[np.isnan(im_dpc)] = 0

    im_dpc = (im_dpc * 255).astype(np.uint8)

    return im_dpc

def overlay_mask_dpc(color_mask, im_dpc):
    # Overlay the colored mask and DPC image
    # make DPC 3-channel
    im_dpc = np.stack([im_dpc]*3, axis=2)
    return (0.75*im_dpc + 0.25*color_mask).astype(np.uint8)
    
def imread_gcsfs(fs,file_path):
    img_bytes = fs.cat(file_path)
    I = imageio.core.asarray(imageio.imread(img_bytes, file_path.split('.')[-1]))
    return I
	
def create_dpc(source, dataset, file_id, flatfield_left, flatfield_right, fs = None):
    dpc_left  = source + '/' + dataset + '/' + '0' + '/' + file_id + '_' + 'BF_LED_matrix_left_half.bmp'
    dpc_right = source + '/' + dataset + '/' + '0' + '/' + file_id + '_' + 'BF_LED_matrix_right_half.bmp'
    if fs != None:
        I_BF_left = imread_gcsfs(fs, dpc_left)
        I_BF_right = imread_gcsfs(fs,dpc_right)
    else:
        I_BF_left = imageio.imread(dpc_left)
        I_BF_right = imageio.imread(dpc_right)
    if len(I_BF_left.shape)==3: # convert to mono if color
        I_BF_left = I_BF_left[:,:,1]
        I_BF_right = I_BF_right[:,:,1]
    I_BF_left = I_BF_left.astype('float')/255
    I_BF_right = I_BF_right.astype('float')/255
    # flatfield correction
    I_BF_left = I_BF_left/flatfield_left
    I_BF_right = I_BF_right/flatfield_right
    I_DPC = generate_dpc(I_BF_left,I_BF_right)

    return I_DPC

