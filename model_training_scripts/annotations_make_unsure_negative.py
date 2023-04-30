import pandas as pd

# Read .csv file as a dataframe
file_path = '~/Desktop/Octopi/data/to-combine/diff3/combined_ann_diff3'
df = pd.read_csv(file_path+ '.csv')

# Replace all empty elements in column 'B' with -1
df.iloc[:,1].fillna(-1, inplace=True)

# replace all 2's with -1
df.iloc[:,1].replace(2, -1, inplace=True)

df.to_csv(file_path.split('_original')[0]  + '.csv', index=False)
