import pandas as pd

# Read .txt file as a dataframe
file_path = '/home/rinni/Desktop/Octopi/data/PAT-070-3_2023-01-22_15-24-28.812821/PAT-070-3_2023-01-22_15-24-28.812821_annotations_FINAL'
df = pd.read_csv(file_path + '.csv')

# Replace all empty elements in column 'B' with -1
df.iloc[:,1].fillna(-1, inplace=True)

# replace all 9's with 2
df.iloc[:,1].replace(9, 2, inplace=True)

df.to_csv(file_path + '_NEW.csv', index=False)
