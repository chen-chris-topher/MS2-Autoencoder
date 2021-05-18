import os
import sys


lower = 70000
upper = lower + 1

script_1 = 'python wget_all_files.py %s %s'
script_2 = 'nextflow run extract_data.nf -c nextflow.config'
script_3 = 'python ./bin/processing.py ../output_nf_1/ ready_array2.npz'
script_4 = 'mv ./bin/big_data.hdf5 ./bin/nominal_%s_%s.hdf5'
script_5 = 'rm -r ./output_nf_1/*'
script_6 = 'rm spectra_data_1/*'


while upper < 100000:
    os.system(script_1 %(lower,upper))
    os.system(script_2)
    os.system(script_3)
    os.system(script_4 %(lower,upper))
    os.system(script_5)
    os.system(script_6)
    
    upper += 1
    lower += 1
    
    break
