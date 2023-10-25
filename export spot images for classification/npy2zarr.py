import xarray as xr
import numpy as np
import zarr

# array = np.load('data.npy')
data = np.random.rand(200, 4, 31, 31)
data = xr.DataArray(data,dims=['t','c','y','x'])
data = data.expand_dims('z')
data = data.transpose('t','c','z','y','x')
# data = (data*255).astype('uint8')

y_dim = data.shape[data.dims.index('y')]
x_dim = data.shape[data.dims.index('x')]

ds = xr.Dataset({'spot_images':data})
ds.spot_images.encoding = {'chunks': (1,1,1,y_dim,x_dim)}
with zarr.ZipStore('spot_images.zip', mode='w') as store:
    ds.to_zarr(store, mode='w')