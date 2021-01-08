import os
import sys

destination_file = './'


with open('./all_file_ftp_list.txt', 'r') as f:
    files = f.read().splitlines()
print(len(files))

#first we get the first 2000
extensions_list = ['.mzxml', '.mzml']
for filename in files[:2000]:
    ext = os.path.splitext(filename)[1]
    
    if ext.lower() in extensions_list: 
        cmd = 'wget ' + filename + ' -P ' + destination_file
        os.system(cmd)
