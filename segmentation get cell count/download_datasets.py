import os
import subprocess

def main():
    # Where to find the data - can be local or remote
    src = 'gs://octopi-malaria-uganda-2022-data'
    # Where to save the CSVs - can be local or remote
    dst = 'results' #'gs://octopi-malaria-data-processing'
    # Get the datasets
    datasets = 'list of datasets.txt'
    
    with open(datasets,'r') as f:
        DATASET_ID = f.read()
        DATASET_ID = DATASET_ID.split('\n')
    for dataset_id in DATASET_ID:
        print(dataset_id)
        savepath = os.path.join(dst, dataset_id, "0")
        os.makedirs(savepath, exist_ok=True)
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{src}/{dataset_id}/0/*_BF_LED_matrix_left_half.bmp", savepath])
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{src}/{dataset_id}/0/*_BF_LED_matrix_right_half.bmp", savepath])
        savepath = os.path.join(dst, dataset_id)
        subprocess.run(["gsutil", "-m", "cp", "-r", f"{src}/{dataset_id}/acquisition parameters.json", savepath])
    return
    
if __name__ == "__main__":
    main()
