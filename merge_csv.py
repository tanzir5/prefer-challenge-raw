from tqdm import tqdm
import pandas as pd
import os

def check_primary_key(df):
    if df['nomem_encr'].is_unique:
        return True
    else:
        raise ValueError("The column 'nomem_encr' is not unique in one of the files.")

# Directory containing the CSV files
directory = './'

# Initialize an empty DataFrame
merged_df = pd.DataFrame()

# Iterate over all files in the directory
for filename in tqdm(os.listdir(directory)):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        # Read the CSV file into a DataFrame
        print(filepath)
        df = pd.read_csv(filepath, usecols=['nomem_encr'])
        check_primary_key(df)
        continue
        if merged_df.empty:
            # Initialize merged_df with the first DataFrame
            merged_df = df
        else:
            # Merge with the existing merged_df on 'nomem_encr'
            merged_df = pd.merge(merged_df, df, on='nomem_encr', how='outer')

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_spreadsheet.csv', index=False)
