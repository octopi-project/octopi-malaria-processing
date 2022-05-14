#!/usr/bin/env python3
import sys
import os
from pathlib import Path
import argparse
import gcsfs
import imageio
import time
import zarr
import numpy as np
import concurrent.futures
import multiprocessing
from skimage import data
from skimage.transform import pyramid_gaussian


"""
To convert a bmp file, run the file as follows:

python zarr-script.py image_name output_directory dimensions

i.e.: 

python bm-to-zarr.py _0_BF_LED_matrix_low_NA.bmp test/bf_low_26x26.zarr 26,26 fl/gs
"""


def update_zarr(lock, img_path, img_name, x, y, dims, grp, n_layers, fs, gs = False):
    print(x, y, ": starting")
    img_bytes = fs.cat(img_path + f"{y}_{x}{img_name}")
    image = imageio.core.asarray(imageio.imread(img_bytes, "bmp"))
    base = image #np.tile(image, (4, 4, 1))\
    if gs:
        gaussian = list(pyramid_gaussian(base, downscale=2, max_layer=n_layers, channel_axis=None))
    else:
        gaussian = list(pyramid_gaussian(base, downscale=2, max_layer=n_layers, channel_axis=-1))
    # print(image.shape)\
    for path, pixels in enumerate(gaussian):
        dim = int(np.around(3000/2**path))
        pixels = np.around(pixels*255)
        with lock:
            if gs: # if statment for greyscale or not
                grp[str(path)][0,0,0, dim*(dims[1]-y-1):dim*(dims[1]-y), dim*(x):dim*(x+1)] = pixels
            else:
                grp[str(path)][0,0,0, dim*(dims[1]-y-1):dim*(dims[1]-y), dim*(x):dim*(x+1)] = pixels[:,:,0]
                grp[str(path)][0,1,0, dim*(dims[1]-y-1):dim*(dims[1]-y), dim*(x):dim*(x+1)] = pixels[:,:,1]
                grp[str(path)][0,2,0, dim*(dims[1]-y-1):dim*(dims[1]-y), dim*(x):dim*(x+1)] = pixels[:,:,2]
    print(x, y, ": done")

    

if __name__ == '__main__':
    # sys.exit(main(sys.argv[1:]))
    args = (sys.argv[1:])
    img_name = args[0]
    zarr_directory = args[1]
    dims = [int(dim) for dim in args[2].split(',')]
    gs = True
    if len(args) == 4:
        gs = not (args[3] == "fl")
    n_layers = 3
    fs = gcsfs.GCSFileSystem(project='soe-octopi',token='whole-slide-20220214-keys.json')
    data_dir = 'octopi-malaria-tanzania-2021-data/U3D_201910_2022-01-11_23-11-36.799392/0/'
    print(img_name, zarr_directory, dims)
    # store = fs.get_mapper('gs://octopi-malaria-whole-slide/zarr-test/' + zarr_directory) #This is not working
    store = zarr.DirectoryStore(zarr_directory)
    grp = zarr.group(store = store, overwrite=True)
    paths = []
    for path in range(n_layers+1):
        grp.create_dataset(str(path), shape=(1, 3, 1, (3000/2**path)*dims[0], (3000/2**path)*dims[1]), dtype='u1')
        paths.append({"path": str(path)})
    print(grp.tree())
    
    m = multiprocessing.Manager()
    lock = m.Lock()

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for x in range(dims[0]):
            for y in range(dims[1]):
                executor.submit(update_zarr, lock, data_dir, img_name, x, y, dims, grp, n_layers, fs, gs)
    
            
    if gs: #different metadate for bright field or flourecense
        image_data = {
            "id": 1,
            "channels": [
                {
                    "color": "FFFFFF",
                    "window": {"start": 0, "end": 256},
                    "label": "Brightness",
                    "active": True,
                }
            ],
            "rdefs": {
                "model": "greyscale",
            },
        }
        
    else:
    
        image_data = {
            "id": 1,
            "channels": [
                {
                    "color": "FF0000",
                    "window": {"start": 0, "end": 256},
                    "label": "Red",
                    "active": True,
                },
                {
                    "color": "00FF00",
                    "window": {"start": 0, "end": 256},
                    "label": "Green",
                    "active": True,
                },
                {
                    "color": "0000FF",
                    "window": {"start": 0, "end": 256},
                    "label": "Blue",
                    "active": True,
                },
            ],
            "rdefs": {
                "model": "color",
            },
        }

    multiscales = [
        {
            "version": "0.1",
            "datasets": paths,
        }
    ]
    grp.attrs["multiscales"] = multiscales
    grp.attrs["omero"] = image_data