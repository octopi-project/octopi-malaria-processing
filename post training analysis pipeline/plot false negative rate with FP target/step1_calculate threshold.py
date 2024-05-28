import pandas as pd
import numpy as np
import os

# Path to the folder containing your CSV files
folder_path = 'neg'

# Initialize an empty list to store the results
results = []

cell_count_df = pd.read_csv('cell count.csv')

# def find_upper_threshold(values,FP_target):
#     # Sort the values in ascending order
#     sorted_values = np.sort(values)
    
#     # Calculate the threshold for the maximum number of values allowed above it
#     max_values_above_threshold = FP_target / 5e6 * len(values)
    
#     # If the calculated max values above threshold is less than 1, set it to 1 since we're dealing with discrete counts
#     max_values_above_threshold = max(1, max_values_above_threshold)
    
#     # Calculate the index for the threshold value
#     threshold_index = len(values) - int(max_values_above_threshold)
    
#     # Ensure the index is within the bounds of the list
#     threshold_index = max(0, min(threshold_index, len(values) - 1))
    
#     # Retrieve the threshold value
#     threshold_value = sorted_values[threshold_index]
    
#     return threshold_value


def find_threshold(numbers, RBC_count , FP_target):
    # Sort the list of numbers
    sorted_numbers = np.sort(numbers)[::-1]
    print(sorted_numbers[:10])

    # Initialize the threshold to None
    threshold = None
    
    # Iterate through the sorted list
    for i, num in enumerate(sorted_numbers):
        # Calculate the normalized count
        normalized_count = (i + 1) / RBC_count
        
        # Check if the normalized count is larger than x/5e6
        if normalized_count > FP_target / 5e6:
            threshold = num
            break  # Found the threshold, exit the loop
    
    return threshold

# Loop through each file in the folder
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        file_path = os.path.join(folder_path, file)
        file_id = file[:-4]

        # get cell count
        cell_count = cell_count_df.loc[cell_count_df['dataset ID'] == file_id, 'Total Count'].values[0]

        # Load the CSV file
        df = pd.read_csv(file_path)

        # Sort the DataFrame based on 'parasite output' in descending order
        df_sorted = df.sort_values(by='parasite output', ascending=False)
        parasite_output_array = df_sorted['parasite output'].to_numpy()

        '''
        # Find the threshold
        if len(df_sorted)/(cell_count) > 5.0/5e6:
            threshold = df_sorted.iloc[4]['parasite output']  # Get the value of the 5th highest 'parasite output'
        else:
            threshold = df_sorted.iloc[-1]['parasite output']  # If there are less than 5 rows, use the smallest 'parasite output'
        '''

        threshold = find_threshold(parasite_output_array, cell_count , 5)

        # Append the file name and threshold to the results list
        results.append([file, threshold])

# Convert the results list to a DataFrame
results_df = pd.DataFrame(results, columns=['File Name', 'Threshold'])

# Display the DataFrame
print(results_df)

# Save the DataFrame to a CSV file
results_df.to_csv('thresholds_by_file.csv', index=False)

# Sort the DataFrame by the 'Threshold' column
results_df_sorted = results_df.sort_values(by='Threshold')

# Import matplotlib for plotting
import matplotlib.pyplot as plt

# Set the size of the plot
plt.figure(figsize=(10, 8))

# Create a bar plot
x_labels = range(1, len(results_df_sorted) + 1)
# plt.bar(results_df['File Name'], results_df['Threshold'])
plt.bar(x_labels, results_df_sorted['Threshold'])

# Add title and labels to the plot
plt.title('Thresholds by File')
plt.xlabel('File Name')
plt.ylabel('Threshold')

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.tight_layout() # Adjust layout to not cut off labels

# Save the plot to a file before showing it
plt.savefig('thresholds_by_file_plot.png', dpi=300)  # Saves the plot as a PNG file with 300 DPI

plt.show()
