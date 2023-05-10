import xarray as xr
import imageio
from tqdm import tqdm
import numpy as np

dataset = xr.open_zarr("spot_images.zip")['spot_images']
for i in tqdm(range(dataset.shape[0])):
    img = dataset[i, :, 0, :, :].to_numpy().transpose(1,2,0)
    img_fluorescence = img[:,:,[2,1,0]]
    img_dpc = img[:,:,3]
    img_dpc = np.dstack([img_dpc,img_dpc,img_dpc])
    imag_overlay = 0.64*img_fluorescence + 0.36*img_dpc
    imageio.imwrite(f"data/{i}_fluorescence.png", img_fluorescence)
    imageio.imwrite(f"data/{i}_overlay.png", imag_overlay)