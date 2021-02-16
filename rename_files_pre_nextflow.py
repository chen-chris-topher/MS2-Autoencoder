import os

target_file = './spectra_data_4'

all_files = os.listdir(target_file)
all_files_2 = [filename.replace(':','') for filename in all_files]
all_files_2 = [filename.replace('__','_') for filename in all_files_2]

all_files = [os.path.join(target_file, item) for item in all_files]
all_files_2 = [os.path.join(target_file,item) for item in all_files_2]

for item1, item2 in zip(all_files,all_files_2):
    os.rename(item1,item2)
