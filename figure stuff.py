import numpy as np
import imageio
import matplotlib.pyplot as plt

img_paths = '/Users/rinnibhansali/Documents/Stanford/Research/Thesis/figures/im_processing_spots/'
full = img_paths + 'full.bmp'
right = img_paths + 'right.bmp'
left = img_paths + 'left.bmp'
fluor = img_paths + 'fluor.bmp'

I_full = imageio.imread(full) / 255.0
I1 = imageio.imread(right) / 255.0
I2 = imageio.imread(left) / 255.0
I_fluor = imageio.imread(fluor) / 255.0

I_dpc = np.divide(I1-I2,I1+I2)
I_dpc = I_dpc + 0.5
I_dpc[I_dpc<0] = 0
I_dpc[I_dpc>1] = 1
I_dpc3 = np.zeros((I_dpc.shape[0],I_dpc.shape[1],3))
I_dpc3[:,:,0] = I_dpc
I_dpc3[:,:,1] = I_dpc
I_dpc3[:,:,2] = I_dpc

I_overlay = 0.64*I_fluor + 0.36*I_dpc3

plt.imshow(I_overlay)
plt.axis('image')
pts = plt.ginput(1)
plt.show()
# 1437 2386 to 1556 2505
# imageio.imwrite(img_paths + 'dpc.png', I_dpc)
imageio.imwrite(img_paths + 'dapi_overlay.png', I_overlay[2386:2505,1437:1556,:])
# imageio.imwrite(img_paths + 'full_cropped.png', I_full[700:900,700:900,:])
