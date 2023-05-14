# Goal: downsample all images by 2x
import numpy as np
from sklearn.model_selection import train_test_split

im_path = 'combined_images'

im = np.load(im_path + '.npy')

im_ds = im[::2, ::2, :]

im_train, im_test = train_test_split(im, test_size=0.1, random_state=16)
im_ds_train, im_ds_test = train_test_split(im_ds, test_size=0.1, random_state=16)

np.save(im_path + '_ds.npy',im_ds)
np.save(im_path + '_train.npy',im_train)
np.save(im_path + '_val.npy',im_test)
np.save(im_path + '_ds_train.npy',im_ds_train)
np.save(im_path + '_ds_val.npy',im_ds_test)