import os

# Replace this with the actual path of your folder
folder_path = 'model output'

# List all csv files in the folder
csv_files = [os.path.splitext(file)[0] for file in os.listdir(folder_path) if file.endswith('.csv')]

# Sort the list of file names
csv_files.sort()

# Write these file names to a text file
with open('list_of_datasets.txt', 'w') as file:
    for filename in csv_files:
        file.write(filename + '\n')

print("Done! The names of csv files are saved in 'list_of_csv_files.txt'")
