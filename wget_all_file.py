import os
import sys
import numpy as np
destination_file = './spectra_data_4'

def clean_file_names():
    all_files = os.listdir(destination_file)
    change_files = []
    for item in all_files:
        break

files = []
with open('./compress_files.txt', 'r') as f:
    for i,line in enumerate(f):
        if i >= 35000 and i < 55000:
    	    files.append(line.strip()) 

exclusion_files = ['MSV000086287', 'MSV000086483', 'MSV000086208', 'MSV000086236']
files = np.unique(files).tolist()
print(len(files))
print("About to pull files")
#first we get the first 6000
extensions_list = ['.mzxml', '.mzml']
for filename in files:
    ext = os.path.splitext(filename)[1]
    if filename.split('/')[3] not in exclusion_files: 
    	if ext.lower() in extensions_list: 
            print(filename)
            new_name = filename.replace('/','_')
            cmd = 'wget -O '+ new_name + ' ' + filename + ' -P %s' %destination_file
            os.system(cmd)
