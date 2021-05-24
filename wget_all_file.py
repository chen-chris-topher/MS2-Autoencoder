import os
import sys
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lower_num', help='lower number of files')
parser.add_argument('upper_num', help='upper number of files')

args = parser.parse_args()

lower = args.lower_num
upper = args.upper_num

destination_file = './spectra_data'

filename = './all_filenames.csv'
df = pd.read_csv(filename)
print(df)
f = df['filepath'].tolist()

files = []

for i,line in enumerate(f):
    if i >= int(lower_num) and i < int(upper_num):
        files.append(line.strip())

files = np.unique(files).tolist()
print(len(files))
print("About to pull files")

exclusion_files =[]
for filename in files: 
    if filename.split('/')[3] not in exclusion_files:
        new_name = filename.replace('/','_').replace(" ","_")
        new_name = destination_file + '/' + new_name
        filename ='ftp://massive.ucsd.edu/' + filename
        cmd = 'wget -O '+ new_name + ' ' + filename
 
        os.system(cmd)

