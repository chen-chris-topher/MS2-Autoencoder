"""
Script takes library spectra, spikes in noise at various levels
and then saves them in dta format for msreduce compatbility.
"""

import os
import sys
import ast
import pandas as pd
import numpy as np
import scipy.sparse as sp
import scipy.stats as stats
from random import randint
from library_search import download_library_spectra, read_ccms 

def add_noise_msreduce(amount):
    rvs = stats.norm(loc=0.10, scale=0.05).rvs
    noise = sp.random(1, 1000, density = amount, data_rvs=rvs)
    noise = noise.toarray()
    
    noise_mz = []
    noise_i = []
     
    for count, value in enumerate(noise[0]):
        if value > 0.0:
            m = count
            digits = randint(10**3, (10**4)-1) / 10000
            m += digits
            i = value

            noise_mz.append(m)
            noise_i.append(i)
     
    return(noise_mz, noise_i)

def write_dta_file(data, ccms_master_list):
    instrument = ['Q-Exactive', 'Q-Exactive Plus', 'Orbitrap']
    mode = ['Positive', 'positive']
    total_files = 0
    count = 0
    ccms_list = []
    for lib_entry in data:

        #make sure this is the same as the lib spectra things
        if lib_entry['spectrum_id'] not in ccms_master_list: 
            continue
        else:
            peaks = lib_entry['peaks_json']
            peaks = ast.literal_eval(peaks)
            
            mz_array = [item[0] for item in peaks]
            inten_array = [item[1] for item in peaks]
            total_files += 1
            
            #used to keep track of which ccms ids we have present
            ccms_list.append(lib_entry['spectrum_id'])
            for county in range(0,21):
                mz_array = [item[0] for item in peaks]
                inten_array = [item[1] for item in peaks]
                
                county = county / 1000
                
                if county != 0:
                    noise_mz, noise_i = add_noise_msreduce(county)
                    
                if len(inten_array) == 0:
                    print("Failed Intensity Array")
                    continue
                
                max_val = np.amax(inten_array)
                inten_array = np.true_divide(inten_array, max_val)
                inten_array[inten_array < 0.05] = 0.0
                inten_array = inten_array.tolist()
                
                if county != 0: 
                    mz_array.extend(noise_mz)
                    inten_array.extend(noise_i)
                
                ccms_id = lib_entry['spectrum_id']
                prec_mz = lib_entry['Precursor_MZ']
               
                mz_array, inten_array = zip(*sorted(zip(mz_array, inten_array)))

                with open("./dta_files/%s.%s.%s.dta" %(ccms_id, ccms_id[-6:],int(county*1000)), "w") as myfile:
                    myfile.write(str(prec_mz) + " " + str(count) + '\n')
                    for m, i in zip(mz_array, inten_array):
                        if i > 0.0:
                            myfile.write(str(m) + ' ' + str(i) + '\n')
                count += 1   

        #we only want 10,000 library spectra in use        
        if total_files > 10000:
            break
    print(total_files)
    with open('./msreduce_dta_files.txt', 'w') as txt_file:
        for line in ccms_list:
            txt_file.write(line + '\n')
def main():
    ccms_filename = './ccms_spectra.txt'
    ccms_list = read_ccms(ccms_filename)[:10000]
    print(len(ccms_list))
    data = download_library_spectra()
    write_dta_file(data, ccms_list)


if __name__ == "__main__":
    main()
