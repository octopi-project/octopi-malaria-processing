import pandas as pd
import os

# Load the thresholds DataFrame
thresholds_df = pd.read_csv('thresholds_by_file.csv')

# Path to the 'pos' folder
pos_folder_path = 'pos'

# Initialize an empty list to store the results
results = []

# Iterate over each file in the 'pos' folder
for pos_file in os.listdir(pos_folder_path):
    if pos_file.endswith('.csv'):
        pos_file_path = os.path.join(pos_folder_path, pos_file)
        # Load the pos file
        pos_df = pd.read_csv(pos_file_path)
        
        # For each row in the thresholds DataFrame, calculate the ratio
        for index, row in thresholds_df.iterrows():
            threshold = row['Threshold']
            # Calculate the ratio of rows with "parasite output" > threshold
            ratio = 1 - sum(pos_df['parasite output'] > threshold) / len(pos_df)
            # Append the results including the pos file name, threshold file name, and the calculated ratio
            results.append([pos_file, row['File Name'], ratio])

# Convert the results list to a DataFrame
ratio_df = pd.DataFrame(results, columns=['Pos File', 'Threshold File', 'Ratio'])

# Save the resulting DataFrame to a CSV file
ratio_df.to_csv('ratio_of_rows_above_threshold.csv', index=False)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assuming 'ratio_df' is the DataFrame you've created in the previous step

# Pivot the DataFrame to create the m x n array
# 'Threshold File' as index, 'Pos File' as columns, and 'Ratio' as values
matrix_df = ratio_df.pivot(index='Threshold File', columns='Pos File', values='Ratio')

# Save the pivoted DataFrame (m x n array) to a CSV file
matrix_df.to_csv('ratio_matrix.csv')

# Plotting
fig, ax = plt.subplots()
# c = ax.pcolor(matrix_df, cmap='Blues', edgecolors='w', linewidths=2)
# c = ax.pcolor(matrix_df, cmap='Reds')
# c = ax.pcolor(matrix_df, vmin=0, vmax=1)
c = ax.pcolor(matrix_df, cmap='Reds', vmin=0, vmax=1)

# # Set the ticks in the middle of each cell
ax.set_xticks(np.arange(matrix_df.shape[1]), minor=False)
# ax.set_yticks(np.arange(matrix_df.shape[0]) + 0.5, minor=False)

# # Want the ticks to be the names of the 'Pos File' and 'Threshold File'
# ax.set_xticklabels(matrix_df.columns, minor=False)
ax.set_xticklabels([col[:7] for col in matrix_df.columns], minor=False)
# ax.set_yticklabels(matrix_df.index, minor=False)

# # Rotate the tick labels for the x-axis for better readability
plt.xticks(rotation=45)

'''
# Generate numerical labels based on the shape of matrix_df
x_labels = range(1, matrix_df.shape[1] + 1)
y_labels = range(1, matrix_df.shape[0] + 1)

# Set the ticks in the middle of each cell
# ax.set_xticks(np.arange(len(x_labels)) + 0.5, minor=False)
# ax.set_yticks(np.arange(len(y_labels)) + 0.5, minor=False)

# Set numerical tick labels
# ax.set_xticklabels(x_labels)
# ax.set_yticklabels(y_labels)
'''

# Add a colorbar to show the ratio values
color_bar = fig.colorbar(c, ax=ax)
color_bar.set_label('False Negative Rate')  # Set the label for the color bar

# Add title and axis labels as needed
plt.title('FNR at FP = 5/ul')
plt.xlabel('Pos slides')
plt.ylabel('Neg slides')

plt.tight_layout()

# Save the plot
plt.savefig('ratio_matrix_plot.png', dpi=300)

# Show the plot
plt.show()
