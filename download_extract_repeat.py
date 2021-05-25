import os
import sys

lower = 1 
upper = 5000

script_1 = 'python wget_all_files.py %s %s'
script_2 = 'nextflow run extract_data.nf -c cluster.config'
script_3 = 'python ./bin/processing.py ./output_nf_1/ ready_array2.npz'
script_4 = 'mv ./big_data.hdf5 ./bin/nominal_%s_%s.hdf5'
script_45 = 'mv ./all_files.txt ./bin/nominal_%s_%s_filenames.txt'
script_5 = 'rm -r ./output_nf_1/*'
script_6 = 'rm spectra_data_1/*'


while upper < 15000:
    os.system(script_1 %(lower,upper))
    os.system(script_2)
    os.system(script_3)
    os.system(script_4 %(lower,upper))
    os.system(script_45 %(lower,upper))
    os.system(script_5)
    os.system(script_6)
    
    upper += 5000
    lower += 5000
    
