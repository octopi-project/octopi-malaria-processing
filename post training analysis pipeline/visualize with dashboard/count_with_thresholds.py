import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

for thresh in tqdm(np.arange(0.9, 1.001, 0.001)):

    print(thresh)

    # Define the path to the folder containing .npy files
    folder_path = '../model output'

    # print("List of datasets to export and infer have been created.")
    with open('list_of_datasets.txt','r') as dataset_file:
        datasets_to_infer = [line.strip() for line in dataset_file.readlines()]

    data = {
        "dataset ID": datasets_to_infer,
        "predicted positive": [-1] * len(datasets_to_infer),
        "predicted negative": [-1] * len(datasets_to_infer)
    }
    all_dataset_prediction_counts = pd.DataFrame(data)

    for dataset_id_0 in datasets_to_infer:
        dataset_id = folder_path + '/' + dataset_id_0
        # print(dataset_id)

        # USER PARAMETERS (optional)
        unsure_ignored = True 

        # intermediate / output paths
        path_csv_annotations_and_predictions = dataset_id + '.csv'

        # Read the CSV file into a DataFrame
        df = pd.read_csv(path_csv_annotations_and_predictions)

        # Calculate pred_pos, pred_neg, and pred_unsure based on your conditions
        pred_pos = len(df[df['parasite output'] > thresh])
        pred_neg = len(df[(df['parasite output'] < thresh)])
        all_dataset_prediction_counts.loc[all_dataset_prediction_counts['dataset ID'] == dataset_id_0, ['predicted positive', 'predicted negative']] = [pred_pos, pred_neg]

        # # Print the counts
        # print("For dataset " + dataset_id)
        # print("pred_pos:", pred_pos)
        # print("pred_neg:", pred_neg)
        # print("pred_unsure:", pred_unsure)

    ## add segmentation stats
    df2 = pd.read_csv('segmentation stats.csv')
    # Renaming the 'Dataset ID' column in df2 to match the 'dataset ID' column in df1 for consistency
    df2 = df2.rename(columns={"Dataset ID": "dataset ID"})
    # Merging the datasets on 'dataset ID'
    merged_df = pd.merge(all_dataset_prediction_counts, df2, on="dataset ID")
    # Calculating the number of positives per (Total Count / 5e6)
    merged_df['Positives per 5M RBC'] = merged_df['predicted positive'] / (merged_df['Total Count'] / 5e6)

    # save
    merged_df.to_csv('count vs threshold/all_dataset_prediction_counts_' + str(thresh)[0:5] + '.csv')