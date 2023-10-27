from interactive_m2unet_inference import M2UnetInteractiveModel as m2u
import gcsfs
import os
import pandas as pd
from tqdm import tqdm
from itertools import product
from scipy.ndimage import label
import imageio
import numpy as np
import json
from utils import *
import cv2

DEBUGGING = False # set true to only process 4 FOVs from each dataset

def main():
    # Get the M2U-Net model
    model_path = 'models/model_70_11.pth'
    use_trt = False
    model = m2u(pretrained_model=model_path, use_trt=use_trt)
    # Get the GCSFS filesystem - set fs = None for local
    gcs_project = 'soe-octopi'
    gcs_token = '/home/octopi-codex/Documents/keys/data-20220317-keys.json'
    fs = gcsfs.GCSFileSystem(project=gcs_project,token=gcs_token)
    # Where to find the data - can be local or remote
    src = 'gs://octopi-malaria-uganda-2022-data'
    # Where to save the CSVs - can be local or remote
    dst = 'results' #'gs://octopi-malaria-data-processing'
    # Options to save masks, overlays
    save_masks = True
    save_overlays = True
    # Get the flatfield correction .npy files - assume they are stored locally
    flatfield_left = np.load('flatfield_left.npy')
    flatfield_right = np.load('flatfield_right.npy')
    # Get the datasets
    datasets = 'list of datasets.txt'
    with open(datasets,'r') as f:
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
    for dataset_id in DATASET_ID:
        print(dataset_id)
        count = run_segmentation(fs, model, src, dst, dataset_id, flatfield_left, flatfield_right, save_masks, save_overlays)
    return

def run_segmentation(fs, model, source, dest, dataset_ID, flatfield_left, flatfield_right, save_masks, save_overlays):
    # Check if our source and destination are remote
    source_remote = False
    source_open = open
    if source[0:5] == 'gs://':
        source_remote = True
        source_open = fs.open
    dest_remote = False
    dest_open = open
    if dest[0:5] == 'gs://':
        dest_remote = True
        dest_open = fs.open
    
    # Look at the source acquisition parameters to get Nx, Ny
    json_filename = source + '/' + dataset_ID + '/acquisition parameters.json'
    with source_open(json_filename, 'r') as f:
    	acquisition_parameters = json.load(f)
    acquisition_parameters['Nx'] = int(acquisition_parameters['Nx'])
    acquisition_parameters['Ny'] = int(acquisition_parameters['Ny'])
    acquisition_parameters['Nz'] = int(acquisition_parameters['Nz'])
    
    if DEBUGGING:
       acquisition_parameters['Nx'] = min(2, acquisition_parameters['Nx'])
       acquisition_parameters['Ny'] = min(2, acquisition_parameters['Ny'])
       acquisition_parameters['Nz'] = min(2, acquisition_parameters['Nz'])
    
    # Check if there is a segmentation_stat.csv in the dest and read it
    csv_filename = dest + '/' + dataset_ID + '/segmentation_stat.csv'
    remote_segmentation_stat_df = None
    try:
        with dest_open(csv_filename, 'r') as f:
            remote_segmentation_stat_df = pd.read_csv(f)
    except:
        pass
    
    # Make a local version of the df
    local_segmentation_stat_df = pd.DataFrame(columns=['FOV_row','FOV_col','count'])
    total_cells = 0
    
    # Loop through each view to check if it has been segmented.
    for x, y, z in tqdm(product(range(acquisition_parameters['Nx']),range(acquisition_parameters['Ny']),range(acquisition_parameters['Nz'])) , total=(acquisition_parameters['Nx'] * acquisition_parameters['Ny'] * acquisition_parameters['Nz'])):
        if type(remote_segmentation_stat_df) != type(None):
            selection = remote_segmentation_stat_df[(remote_segmentation_stat_df['FOV_row'] == y) & (remote_segmentation_stat_df['FOV_col'] == x)]
            # If we have exactly one entry, concat it onto our local and move on to the next iter
            if len(selection) == 1:
                local_segmentation_stat_df = pd.concat([local_segmentation_stat_df, selection], ignore_index=True)
                continue
        
        # If we didn't continue, do the segmentation ourselves using M2U-Net
        file_id = f"{y}_{x}_{z}"
        I_DPC = create_dpc(source, dataset_ID, file_id, flatfield_left, flatfield_right, fs = fs)
        # Segment DPC image
        result = model.predict_on_images(I_DPC)
        threshold = 0.5
        mask = (255*(result > threshold)).astype(np.uint8)
        # Store segmentation mask
        if save_masks:
            mask_path = dest + '/' + dataset_ID + '/segmentation_mask_binary/mask_' + file_id + '.bmp'
            if dest_remote == False:
                os.makedirs(dest + '/' + dataset_ID + '/segmentation_mask_binary/', exist_ok=True)
            with dest_open(mask_path, 'wb') as f:
                if dest_remote == False:
                    imageio.imwrite(f.name, mask, format='bmp')
                else:
                    imageio.imwrite(f, mask, format='bmp')
        # Get number of cells, store to DF
        labeled_mask, n_cells = label(mask)
        total_cells += n_cells
        local_segmentation_stat_df = pd.concat([pd.DataFrame([[y, x, n_cells]], columns=local_segmentation_stat_df.columns), local_segmentation_stat_df], ignore_index=True)
        # Store overlay
        if save_overlays: 
            overlay = overlay_mask_dpc(colorize_mask(labeled_mask), I_DPC)
            overlay_path = dest + '/' + dataset_ID + '/segmentation_mask_binary/overlay_' + file_id + '.bmp'
            if dest_remote == False:
                os.makedirs(dest + '/' + dataset_ID + '/segmentation_mask_binary/', exist_ok=True)
            with dest_open(overlay_path, 'wb') as f:
                if dest_remote == False:
                    imageio.imwrite(f.name, overlay, format='bmp')
                else:
                    imageio.imwrite(f, overlay, format='bmp')
        
    # Save DF - overwrite
    totalcell_filename = dest + '/' + dataset_ID + '/total_cells.txt'
    
    with dest_open(csv_filename, 'wb') as f:
        local_segmentation_stat_df.to_csv(f, encoding='utf-8', index=False)
    with dest_open(totalcell_filename, 'wb') as f:
        f.write(str(total_cells).encode('utf-8'))
        
    return local_segmentation_stat_df
 
def colorize_mask(labeled_mask):
    # Color them
    colored_mask = np.array((labeled_mask * 83) % 255, dtype=np.uint8)
    colored_mask = cv2.applyColorMap(colored_mask, cv2.COLORMAP_HSV)
    # make sure background is black
    colored_mask[labeled_mask == 0] = 0
    return colored_mask

def overlay_mask_dpc(color_mask, im_dpc):
    # Overlay the colored mask and DPC image
    # make DPC 3-channel
    im_dpc = np.stack([im_dpc]*3, axis=2)
    return (0.75*im_dpc + 0.25*color_mask).astype(np.uint8)
    
if __name__ == '__main__':
    main()
