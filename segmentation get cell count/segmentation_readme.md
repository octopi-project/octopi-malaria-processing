# M2U-Net Segmentation (segmentation_get_cell_count.py)

## Usage Guide

### Inputs and Outputs

This script reads Octopi-generated red blood cell scans and segments them. For each dataset, the script outputs the number of cells seen in each dataset and a CSV with the number of cells seen in each field of view. The script optionally saves binary masks and overlays from the segmentation.

### Install Requirements

Run `pip install -r requirements.txt` to install the requirements. We recommend doing this in a fresh virtual environment (venv, conda, or similar).

### Set Parameters

* `model_path`: String, path to M2U*Net model. The file must be in the same directory as a `config.json` with model metadata.
* `use_trt`: Boolean, set to True if using a TensorRT model. Set to False if using a PyTorch model.
* `gcs_project`: String, name of GCS project if reading from or writing to cloud storage. Leave empty if running locally.
* `gcs_token`: String, path to json key for accessing GCS data. Leave empty if running locally.
* `src`: String, path to source data. Google Cloud bucket URI if remote (must start with `gs://`) or path to a local directory (can be relative or global path).
  - Note: This code downloads images from Google Cloud individually - this process is slow! If you have the extra disc space, it would be faster to download the data beforehand and run this script locally on the downloaded data. Run the following commands to download individual datasets using `gsutil`:
    - `gsutil -m cp -r 'gs://{src_remote}/{dataset}/0/*_BF_LED_matrix_left_half.bmp' './{dataset}/0/'`: Download the left-illuminated images
    - `gsutil -m cp -r 'gs://{src_remote}/{dataset}/0/*_BF_LED_matrix_right_half.bmp' './{dataset}/0/'`: Download the right-illuminated images
    - `gsutil -m cp 'gs://{src_remote}/{dataset}/acquisition parameters.json' './{dataset}'`: Download the acquisition parameters
  - You can also use the script `download_datasets.py` to do this for you.
* `dst`: String, path to save data. Google Cloud bucket URI if remote (must start with `gs://`) or path to a local directory (can be relative or global path).
* `save_masks`: Boolean, set to True to save binary masks.
* `save_overlays`: Boolean, set to True to save masks overlaid on the cell data.
* `datasets`: String, path to file listing the datasets to segment. Must be local.
