## Data/Input
- list of datasets.txt
- GCS json key
- processed scans stored on GCS, including spot_data_raw.csv
- optionally, spot_data_selected csv file stored locally

## Scripts
- export_spot_images.py: generate a zip store zarr and upload it to GCS, with a spot to location mapping.csv
- download_spot_images.py: download the zip store zarr
- zarr_to_png.py: convert the zip store zarr to individual png files
