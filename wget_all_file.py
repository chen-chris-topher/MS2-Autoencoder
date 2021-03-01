import os
import sys
import numpy as np
import pandas as pd

destination_file = './spectra_data_1'

filename = './all_filenames.csv'
df = pd.read_csv(filename)
print(df)

f = df['filepath'].tolist()

files = []

for i,line in enumerate(f):
    if i >= 0 and i < 22000:
        files.append(line) 

files = np.unique(files).tolist()
print(len(files))
print("About to pull files")

extensions_list = ['.mzxml', '.mzml']
for filename in files:
    ext = os.path.splitext(filename)[1]
    
     
    if ext.lower() in extensions_list: 
        print(filename)
        new_name = filename.replace('/','_')
        filename = 'ftp://massive.ucsd.edu/' + filename
        cmd = 'wget -O '+ destination_file + "/" + new_name + ' ' + filename 
        os.system(cmd)
