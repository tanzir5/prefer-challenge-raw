"Remove nohouse variables and transform them into an indicator if nohouse is there or not"

import pandas as pd 
from tqdm import tqdm
import re
import subprocess
import numpy as np 

rootdir = "/home/flavio/OneDrive/NLeSC/Projects/2024_life2vec/stakeholders/CBS/imports/2024-05-17_LISS_PreFer/data/"

source = "liss_merged_spreadsheet.csv"
target = "liss_merged_nohouse_cleaned.csv"
chunk_size = 100


source_file = rootdir + source
target_file = rootdir + target


df_check = pd.read_csv(source_file, nrows=2)
crit = [x for x in df_check.columns if "nohouse" in x]

cmd = ["wc", "-l", source_file]
res = subprocess.run(cmd, check=True, capture_output=True)
total_lines = int(res.stdout.decode().split()[0])

total_chunks = (total_lines - 1) // chunk_size + 1

df_iterator = pd.read_csv(rootdir + source, chunksize=chunk_size, low_memory=False)

first_chunk = True 
for chunk in tqdm(df_iterator, total=total_chunks):
    for col in crit: 
        newcol = re.sub("_nohouse_encr", "_has_hhid", col)
        chunk[col] = np.where(
            chunk[col].isna(),
            0, 1
        )
        chunk.rename(columns={col: newcol})


    if first_chunk:
        chunk.to_csv(target_file, index=False)
        first_chunk = False 
    else:
        chunk.to_csv(target_file, mode="a", index=False, header=False)



