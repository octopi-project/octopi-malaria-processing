# Goal: downsample all images by 2x
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

im_path = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images'
ann_path = '/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann'

im = np.load(im_path + '.npy')
annotation_df = pd.read_csv(ann_path + '.csv', index_col='index').sort_index()
anns = annotation_df['annotation'].values.squeeze()

im_ds = im[:, :, ::2, ::2]

indices = np.random.choice(len(im), len(im), replace=False)
data = im[indices,:,:,:]
anns = anns[indices]
data_ds = im_ds[indices,:,:,:]

im_train, im_val = np.split(data, [int(0.7*len(im))])
ann_train, ann_val = np.split(anns, [int(0.7*len(anns))])
im_ds_train, im_ds_val = np.split(data_ds, [int(0.7*len(im_ds))])

train_annotations = pd.DataFrame({'annotation':ann_train})
train_annotations.index.name = 'index'
test_annotations = pd.DataFrame({'annotation':ann_val})
test_annotations.index.name = 'index'

np.save(im_path + '_ds_all.npy',im_ds)
np.save(im_path + '.npy',im_train)
np.save(im_path + '_val.npy',im_val)
np.save(im_path + '_ds.npy',im_ds_train)
np.save(im_path + '_ds_val.npy',im_ds_val)

train_annotations.to_csv(ann_path + '.csv')
test_annotations.to_csv(ann_path + '_val.csv')




# im_t = np.load('/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_all.npy')
# im_v = np.load('/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_all_val.npy')
# ann_t = pd.read_csv('/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_all.csv',index_col='index')
# ann_v = pd.read_csv('/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_all_val.csv',index_col='index')

# im_all = np.concatenate((im_t,im_v),axis=0)
# ann_all = pd.concat([ann_t,ann_v],ignore_index=True)
# ann_all = ann_all.sort_index()

# ann_all.to_csv('/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_ann_all1.csv')
# np.save('/media/rinni/Extreme SSD/Rinni/to-combine/sorted_images/combined_images_all1.npy',im_all)


# np.random.seed(42)  # set the seed to make the random permutation reproducible
# indices = np.random.permutation(im.shape[0])
# indices = np.sort(indices)
# train_indices = indices[:int(0.9*len(indices))]
# test_indices = indices[int(0.9*len(indices)):]

# im_train = im[train_indices]
# im_test = im[test_indices]

# im_ds_train = im_ds[train_indices]
# im_ds_test = im_ds[test_indices]

# train_annotations = annotation_df.iloc[train_indices]
# train_annotations = train_annotations['annotation'].values.squeeze()
# test_annotations = annotation_df.iloc[test_indices]
# test_annotations = test_annotations['annotation'].values.squeeze()