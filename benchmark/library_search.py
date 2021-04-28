"""
Takes in all + mode Q-Exactive Library Search data and
1. Extracts it into the hdf5 format
2. Spikes in low level sparse noise
3. Attempts to remove noise via Denoising Model
4. Reformats into mgf for library search
5. Runs library search at GNPS 
"""
from spoof_noise_recovery import denoise, add_noise, mirror_plots, cosine_score, bin_reduce_normalize

import sys
import ast
import math
import json
import h5py
import copy
import requests
from random import randint
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyteomics import mgf
from scipy.spatial.distance import cosine
from scipy.stats import binned_statistic

def fetch_compound_name(ccms_list, data):
    
    ccms_l = []
    name = []
    inchi = []
    smiles =[] 
    for lib_entry in data:
        ccms_id = lib_entry['spectrum_id']
        if ccms_id in ccms_list:
            compound_name = lib_entry['Compound_Name']
                        
            ccms_l.append(ccms_id)
            inchi.append(lib_entry['INCHI'])
            smiles.append(lib_entry["Smiles"])
            name.append(compound_name)
    
    compound_name_df = pd.DataFrame({'SpectrumID_ccms':ccms_l, 'Compound_Name_ccms':name, 'Smiles_ccms':smiles, 'INCHI_ccms':inchi})
    compound_name_df.to_csv("compound_names_2.csv")
  
def download_library_spectra():
    """
    Function queries for all library spectra from GNPS 
    that are collected in positive mode on a QE instrument
    and returns it as a dictionary.
    """
    
    peaks_url = 'https://gnps-external.ucsd.edu/gnpslibraryformattedwithpeaksjson'

    response = requests.get(peaks_url)
    data = response.json()

    return(data)

def extract_library_spectra(data):
    """
    Function takes in GNPS library data and yeilds binned intensitys 
    and CCMS Lib IDs for tracking spectra.
    """
 
    instrument = ['Q-Exactive', 'Q-Exactive Plus', 'Q-Exactive Plus Orbitrap Res 14k', 'Q-Exactive Plus Orbitrap Res 70k']
    mode = 'Positive' 

    
    #loop over entire library
    for lib_entry in data: 
        if lib_entry['Instrument'] == 'Orbitrap' and (lib_entry['Ion_Mode'] == mode or lib_entry['Ion_Mode'] == 'positive'):     
            ccms_id = lib_entry['spectrum_id']
            peaks = lib_entry['peaks_json']
            peaks = ast.literal_eval(peaks)
            
            mz_array = [item[0] for item in peaks]
            inten_array = [item[1] for item in peaks]
             
            #bin the sepctra nominally
            binned_intensity, binned_mz, _ = binned_statistic(mz_array, inten_array, statistic='sum', bins=2000, range=(0, 2000))
                       

            #normalize and drop low values
            inten_max = np.amax(binned_intensity)
            binned_intensity = np.true_divide(binned_intensity, inten_max)
            binned_intensity[binned_intensity < 0.05] = 0.0
            
            yield(binned_intensity, ccms_id)
    
def make_hdf5_file():
    """
    Function makes hdf5 file for storage with metadata
    column for CCMS Lib ID storage.
    """
    name = 'lib_spectra_3.hdf5'

    with h5py.File(name, 'w') as f:
        dataset = f.create_dataset('lib_spectra', shape=(1, 2000), maxshape=(None, 2000),compression='gzip')
        f.close()

def write_to_hdf5(data):
    lib_id_list = []
    name = 'lib_spectra_3.hdf5'

    for binned_intensity, ccms_id in extract_library_spectra(data):
        if np.isnan(binned_intensity).any():
            continue
        if np.count_nonzero(binned_intensity) == 0:
            continue
        lib_id_list.append(ccms_id)
             
        with h5py.File(name, 'a') as f:
            dataset = f['lib_spectra']
            size = dataset.shape
            curr_len = size[0]
            new_len = curr_len + 1

            dataset.resize((new_len, 2000))
            dataset[curr_len, :] = binned_intensity
            
            if dataset.shape[0] % 1000 == 0:
                print(dataset.shape)

    
    with open('ccms_spectra_3.txt', 'w') as f:
        for item in lib_id_list:
            f.write("%s\n" % item)

def check_file():
    name = 'lib_spectra.hdf5'

    with h5py.File(name, 'r') as f:
        dataset = f['lib_spectra']
        print(list(dataset[1]))
        print(dataset.shape)
    
    with open('ccms_spectra.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
    print(len(lines))

def read_hdf5():
    name = 'lib_spectra_3.hdf5'
    f = h5py.File(name, 'r')
    data = f.get('lib_spectra').value
    print("Lib Data Shape ", data.shape)
    return(data)

def reformat_mgf(data, full_data, ccms_list, paramFilename):
    from pyteomics import mgf
    count =1 
    mz_array = range(0,2000)
    spectra_list = []

    #go trhough the ccms_list one at a time
    for ccms, i_data in zip(ccms_list, data):
        for a in full_data:
            if a['spectrum_id'] == ccms:
                item = a
                intensity_array = i_data 
            
                peaks = item['peaks_json']
                peaks = ast.literal_eval(peaks)
            
                file_mz_array = [item[0] for item in peaks]
                file_inten_array = [item[1] for item in peaks]
                prec = max(file_inten_array) 
                file_floor_mz = [math.floor(item) for item in file_mz_array] #this should equal bin

                new_mz = []
                new_inten = []
                 
                for m, i in zip(mz_array, intensity_array):
                    #we don't predict no peak
                    if float(i) > 0.0:

                        #if it exists in the original spectra
                        if m in file_floor_mz:
                            #find all mz value location in file data
                            places = [i for i, x in enumerate(file_floor_mz) if x == m]        
                            #print(places)
                            #find the associated intensity values
                            intens = [file_inten_array[i] for i in places]
                            #print(intens)
                            
                            large_inten_loc = intens.index(max(intens))
                            #print(intensity_array, mz_array) 
                            actual_mz = file_mz_array[places[large_inten_loc]] 
                            #print(actual_mz, i)
                            new_mz.append(round(actual_mz,4))    
                            new_inten.append(i * prec)
                            #print(m, actual_mz)
                        
                        #if it's added noise
                        else:                            
                            number_gen = random.randint(1000,9999)
                            mz_gen = (number_gen / 10000) + m   
                            new_mz.append(round(mz_gen,4))
                            new_inten.append(i * prec)
                        
                filename = item['source_file'] 
                mz = item['Precursor_MZ']
                rt = 0.0
                charge = item['Charge']
                intensity = 0.0
                              
                parameters = {"FEATURE_ID": count, 'FILENAME': filename, 'PEPMASS' : str(mz), "SCANS": count, "CCMSLIB": item['spectrum_id'], "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN":prec}
                spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
               
                spectra_list.extend(copy.deepcopy(spectra))
                 
                count += 1
                if count % 1000 == 0:
                    print(count)
        
    mgf.write(spectra = spectra_list, output = "./%s" %paramFilename, write_charges = False, use_numpy = True)    

def special_mirror_plots(spectra_1, mz_1, spectra_2):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
    
    spectra_1 = np.array(spectra_1)
    spectra_2 = np.array(spectra_2)

    mz_2 = np.arange(0,2000)
    spectra = []
 
    max_val = np.amax(spectra_1, axis=0)
    spectra_1 = np.true_divide(spectra_1, max_val)

    spectra.append(sus.MsmsSpectrum(identifier=0, precursor_mz=0, precursor_charge=0, mz=mz_1, intensity=spectra_1))
    spectra.append(sus.MsmsSpectrum(identifier=0, precursor_mz=0, precursor_charge=0, mz=mz_2, intensity=spectra_2))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()

def read_ccms():
    ccms_list = []
    with open('ccms_spectra_3.txt') as hf:
        for line in hf:
            ccms_list.append(line.strip()) 
    return(ccms_list)

def count_original_peaks(ccms_list, lib):
    peak_count = []
    for item in lib:
        peak_count.append(np.count_nonzero(item))
        
    df = pd.DataFrame({'peak_count':peak_count})
    df.to_csv("peak_count_2.csv")

def count_added_peaks(ccms_list, lib, noisy):
    import seaborn as sns
    added_peaks = []
    for item1, item2 in zip(lib, noisy):
        item1 = np.count_nonzero(item1)
        item2 = np.count_nonzero(item2)
        added_peaks.append(item2 - item1)
    
    df = pd.DataFrame({'added_peaks':added_peaks})
    df.to_csv('added_peaks_2.csv')
def filter_data(data, ccms_list):
    new_data = []
    for item in data:
        if item['spectrum_id'] in ccms_list:
            new_data.append(item)
        
    return(new_data)
def main():
    
    #make_hdf5_file()
    data = download_library_spectra()
    #write_to_hdf5(data)
    #sys.exit(0) 
    #check_file()

    lib_data = read_hdf5()[:2000]
    lib_data = bin_reduce_normalize(lib_data)
   

    final_lib = lib_data

    for i in range(1,20):
        final_lib = np.concatenate((final_lib, lib_data))
    
    noisy_peaks = np.load('./noisy_peak_spoof_master_2.npy')
    denoised_data = np.load('./predictions_spoof_master_2.npy') 
    """
    mset = set(range(0,2000))
    first_pass = True
    for count, (n, d) in enumerate(zip(noisy_peaks, denoised_data)):
        a = np.nonzero(n)[0].tolist()
        nindex = set(a)
        oindex = mset - nindex
        zeroi = list(oindex)
        d[zeroi] = 0.0
        denoised_data[count] = d
            
    np.save('./denoised_processed_data_2.npy', denoised_data)
    """
    denoised_data = np.load('./denoised_processed_data_2.npy')
         
    ccms_list = read_ccms()[:2000]
    og_ccms = ccms_list
    fetch_compound_name(ccms_list, data)
    
    final_lib = final_lib
    
    print("Final Lib ", final_lib.shape)
    print("Denoised Data ",denoised_data.shape)
    print("Noisy Peaks ", noisy_peaks.shape)
    ccms_list = ccms_list * 20
    ccms_list = ccms_list
    print("CCMS List ", len(ccms_list))
    
    count_original_peaks(ccms_list, final_lib)
    count_added_peaks(ccms_list, final_lib,noisy_peaks) 
    sys.exit(0)
    #mirror_plots(final_lib[94], denoised_data[94].T)
     
    #print(1-cosine(final_lib[1155], denoised_data[1155].T))

    #cosine_score(denoised_data, final_lib, noisy_peaks)
    #special_mirror_plots(spectra_1, mz_1, spectra_2)        
    print("Before reformatting mgf")    
    data = filter_data(data, og_ccms)
    print(len(data))
    reformat_mgf(denoised_data, data, ccms_list, 'denoised_40.12.mgf')
    reformat_mgf(final_lib, data, ccms_list, 'lib_data_40.12.mgf')
    reformat_mgf(noisy_peaks, data, ccms_list, 'noisy_40.12.mgf')
    launch_lib_search()

if __name__ == "__main__":
    main()

