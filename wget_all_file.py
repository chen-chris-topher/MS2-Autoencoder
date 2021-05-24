import os
import sys
import numpy as np
import pandas as pd
<<<<<<< HEAD

destination_file = './spectra_data_1'
=======
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('lower_num', help='lower number of files')
parser.add_argument('upper_num', help='upper number of files')

args = parser.parse_args()

lower = args.lower_num
upper = args.upper_num

destination_file = './spectra_data'
>>>>>>> b9d957a7a263a1a4b4a2a09c56a0348c680f8fa6

filename = './all_filenames.csv'
df = pd.read_csv(filename)
print(df)
<<<<<<< HEAD

=======
>>>>>>> b9d957a7a263a1a4b4a2a09c56a0348c680f8fa6
f = df['filepath'].tolist()

files = []

for i,line in enumerate(f):
<<<<<<< HEAD
    if i >= 0 and i < 22000:
        files.append(line) 
=======
    if i >= int(lower_num) and i < int(upper_num):
        files.append(line.strip())
>>>>>>> b9d957a7a263a1a4b4a2a09c56a0348c680f8fa6

files = np.unique(files).tolist()
print(len(files))
print("About to pull files")

<<<<<<< HEAD
extensions_list = ['.mzxml', '.mzml']
for filename in files:
    ext = os.path.splitext(filename)[1]
    
     
    if ext.lower() in extensions_list: 
        print(filename)
        new_name = filename.replace('/','_')
        filename = 'ftp://massive.ucsd.edu/' + filename
        cmd = 'wget -O '+ destination_file + "/" + new_name + ' ' + filename 
        os.system(cmd)
=======
exclusion_files =[]
for filename in files: 
    if filename.split('/')[3] not in exclusion_files:
        new_name = filename.replace('/','_').replace(" ","_")
        new_name = destination_file + '/' + new_name
        filename ='ftp://massive.ucsd.edu/' + filename
        cmd = 'wget -O '+ new_name + ' ' + filename
 
        os.system(cmd)

>>>>>>> b9d957a7a263a1a4b4a2a09c56a0348c680f8fa6
