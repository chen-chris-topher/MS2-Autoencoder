"""
Script takes a target compound and plots all noisy, denoised, and library 
spectra.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus


from library_search import read_hdf5
from spoof_noise_recovery import bin_reduce_normalize


lib_data = read_hdf5()
lib_data = bin_reduce_normalize(lib_data)
final_lib = lib_data
for i in range(1,20):
    final_lib = np.concatenate((final_lib, lib_data))

noisy_peaks = np.load('./noisy_peak_spoof_master.npy')
denoised = np.load('denoised_processed_data.npy')


all_indices_plot = range(191, 12276, 636)

added_peaks = pd.read_csv('added_peaks.csv')
added_peaks_list = added_peaks['added_peaks'].tolist()

for i in all_indices_plot:
    add_peaks = added_peaks_list[i]
    if add_peaks == 0:
        continue

    n_spec = noisy_peaks[i][:1000]
    d_spec = denoised[i][:1000]
    l_spec = final_lib[i][:1000]
    spec_list = [n_spec, d_spec, l_spec]
    mz_og = [item for item,x in enumerate(l_spec) if x > 0] 
    og_peaks = ['purple' if x > 0 else 'orange' for x in l_spec]
    print(mz_og)
    #sys.exit()
    for count,spectra in enumerate(spec_list):
        if count == 0:
            title = 'Noisy'
        elif count == 1:
            title = 'Denoised'
        else:
            title = 'Library'

        precursor_mz = 0.0
        precursor_charge = 0
        mz = range(0,1000)
        
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.yaxis.grid(color='gray', linestyle='dashed')
        #f title == 'Noisy' or title == 'Library':
        ax.bar(x=mz, height=spectra, width = 3, color = og_peaks)
        

        plt.title("2'-DEOXYADENOSINE 5'-DIPHOSPHATE, %s Noise Peaks, %s" %(add_peaks, title))
        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        #plt.show()
         
        fig.savefig("./temp_figs/%s_%s.png" %(add_peaks, title))
        plt.close()
        
