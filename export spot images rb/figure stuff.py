import numpy as np
import imageio

img_paths = '/Users/rinnibhansali/Documents/Stanford/Research/Thesis/figures/led array/'
full = img_paths + 'full.png'
right = img_paths + 'right.png'
left = img_paths + 'left.png'

I_full = imageio.imread(full)
I1 = imageio.imread(right) / 255.0
I2 = imageio.imread(left) / 255.0

I_dpc = np.divide(I1-I2,I1+I2)
I_dpc = I_dpc + 0.5
I_dpc[I_dpc<0] = 0
I_dpc[I_dpc>1] = 1

imageio.imwrite(img_paths + 'dpc.png', I_dpc)
imageio.imwrite(img_paths + 'dpc_cropped.png', I_dpc[500:900,500:900,:])
imageio.imwrite(img_paths + 'full_cropped.png', I_full[500:900,500:900,:])
