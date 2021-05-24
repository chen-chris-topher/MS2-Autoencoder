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


def fetch_molecular_formula(ccms_list, data):
    """
    Parameters :
        ccms_list (list) : a list of all ccms ids to get a 
        molecular formula for

        data (list) : a list of dictionaries representing all library
        spectra at GNPS
    
    Function saves a csv with the molecular formula for the ccms ids of interest.
    Used for sirius implementation.
    """
    ccms_l = []
    formula = []

    for lib_entry in data:
        ccms_id = lib_entry['spectrum_id']
        
        #execute if its a part of our target list
        if ccms_id in ccms_list:
            compound_inchi = lib_entry['Formula_inchi']
            compound_smiles = lib_entry['Formula_smiles']
            
            if len(compound_inchi) > 0:
                formula.append(compound_inchi)
                ccms_l.append(ccms_id)
            
            elif len(compound_smiles) > 0:
                formula.append(compound_smiles)
                ccms_l.append(ccms_id)
    
    
    compound_name_df = pd.DataFrame({'SpectrumID_ccms':ccms_l, 'Formulas':formula})
    compound_name_df.to_csv("molecular_formulas.csv")
 

def fetch_compound_name(ccms_list, data):
    """
    Parameters :
        ccms_list (list) : a list of all ccms ids to get a 
        molecular formula for

        data (list) : a list of dictionaries representing all library
        spectra at GNPS
    
    Function saves a csv with the compound names, smiles, and inchis for the 
    ccms ids of interest. Used to assess the results post-library search.
    """
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
    compound_name_df.to_csv("compound_names_1_10000.csv")
  
def download_library_spectra():
    """
    Returns:
        data (list) : list of dictionaries representing all library
        spectra at GNPS

    Function queries for all library spectra from GNPS 
    and returns it as a list.
    """
    
    peaks_url = 'https://gnps-external.ucsd.edu/gnpslibraryformattedwithpeaksjson'
    response = requests.get(peaks_url)
    data = response.json()
    return(data)

def extract_library_spectra(data, ccms_exclusion):
    """
    Parameters:
        data (list) : list of dictionaries representing all library
        spectra at GNPS

    Function takes in GNPS library data and yeilds binned, normalized, and lowest
    level noise removed  spectra vectors and CCMS Lib IDs for tracking spectra.
    """
 
    instrument = ['Q-Exactive', 'Q-Exactive Plus', 'Q-Exactive Plus Orbitrap Res 14k', 'Q-Exactive Plus Orbitrap Res 70k', 'Orbitrap']
    mode = ['Positive', 'positive'] 
    
    #loop over entire library
    for lib_entry in data: 
        if (lib_entry['Instrument'] == 'Orbitrap' or lib_entry['Instrument'] in instrument) and (lib_entry['Ion_Mode'] in mode):                 
            ccms_id = lib_entry['spectrum_id']
            
            #if we provide an exlcusion list we need the ccms id to not be in it
            if ccms_exclusion is not None:
                if ccms_id in ccms_exclusion:
                    continue

            peaks = lib_entry['peaks_json']
            peaks = ast.literal_eval(peaks)
            
            mz_array = [item[0] for item in peaks]
            inten_array = [item[1] for item in peaks]
             
            #bin the spectra nominally
            binned_intensity, binned_mz, _ = binned_statistic(mz_array, inten_array, statistic='sum', bins=2000, range=(0, 2000))
                       
            #normalize and drop low values
            inten_max = np.amax(binned_intensity)
            binned_intensity = np.true_divide(binned_intensity, inten_max)
            binned_intensity[binned_intensity < 0.05] = 0.0
            
            yield(binned_intensity, ccms_id)
    
def make_hdf5_file(name):
    """
    Parameters:
        name (str) : the name of the hdf5 file to create

    Function makes hdf5 file for storage with metadata
    column for CCMS Lib ID storage.
    """
    
    with h5py.File(name, 'w') as f:
        dataset = f.create_dataset('lib_spectra', shape=(1, 2000), maxshape=(None, 2000),compression='gzip')
        f.close()

def write_to_hdf5(data, name, ccms_exclusion=None):
    """
    Parameters :
        data (list) : data downloaded directly from GNPS, a list of dictionary 
        containing all information about all library spectra available

        name (str) : name of the hdf5 file to write to
  
    Function takes all library data, sorts out into what fits the pre-defined search
    criteria, bins it, normlizes it, removes lowest-level noise, and save it to an hdf5
    file. CCMS ids are saved to a text file in the same order.
    """
    lib_id_list = []
    
    for binned_intensity, ccms_id in extract_library_spectra(data, ccms_exclusion):
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
    
    #the order in which library spectra appear in our validation data
    with open('ccms_spectra.txt', 'w') as f:
        for item in lib_id_list:
            f.write("%s\n" % item)

def check_file(name):
    """
    Parameters:
        name (str) : the name of the hdf5 file to test

    Function is used as a sanity check to make sure that the shape 
    of the saved data and the number of corresponding ids we have are
    the same length.
    """

    with h5py.File(name, 'r') as f:
        dataset = f['lib_spectra']
        print("Dataset %s Shape" %name, dataset.shape)
    
    with open('ccms_spectra.txt', 'r') as f:
        lines = [line.rstrip() for line in f]
        print("CCMS ID List Length", len(lines))

def read_hdf5(name, num_used_spectra):
    """
    Parameters:
        name (str) : the name of the hdf5 file to test
        
        num_used_spectra (int) : the slice value for the hdf5 file
    Returns:
        data (numpy matrix) : the loaded hdf5 data
        
    Function is used to load hdf5 data into memory.
    """
    
    f = h5py.File(name, 'r')
    data = f.get('lib_spectra').value[:num_used_spectra]
    return(data)

def reformat_mgf(data, full_data, ccms_list, paramFilename): 
    """


    """
    
    
    count =1 
    mz_array = range(0,2000)
    spectra_list = []
    pass_num = 0
    
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
                file_inten_array = [item/prec for item in file_inten_array]
                 
                file_floor_mz = [math.floor(item) for item in file_mz_array] #this should equal bin

                new_mz = []
                new_inten = []
                for m, i in zip(mz_array, intensity_array):
                    #we don't predict no peak
                    if float(i) > 0.0:
                         
                        #if it exists in the original spectra
                        if m in file_floor_mz:
                            #find all mz value location in file data
                            places = [a for a, x in enumerate(file_floor_mz) if x == m]        
                            
                            intens_og = [file_inten_array[a] for a in places]
                            #find the associated intensity values
                            intens = [a for a in places if file_inten_array[a] > 0.005]
                            if len(intens) == 0:
                                continue
                            
                            #actual m/z
                            actual_mz = [round(file_mz_array[a],4) for a in intens]
                            og_inten = [round(file_inten_array[a],4) for a in intens]
                            
                            og_sum = max(og_inten)
                            og_inten_part = [a/og_sum for a in og_inten]
                            
                            og_inten_split = [(a*i)*prec for a in og_inten_part] #split the predicted inten between em
                            new_inten.extend(og_inten_split)
                            new_mz.extend(actual_mz)
                             
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
                              
                parameters = {"FEATURE_ID": count, 'FILENAME': filename, 'PEPMASS' : str(mz), "SCANS": count, "CCMSLIB": ccms, "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN":prec}
                spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
               
                spectra_list.extend(copy.deepcopy(spectra))
                
                if count % 1000 == 0:
                    print(count)
                count += 1


    mgf.write(spectra = spectra_list, output = "./%s.mgf" %(paramFilename), write_charges = False, use_numpy = True)    
                       
def read_ccms(name):
    ccms_list = []
    with open(name, 'r') as hf:
        for line in hf:
            ccms_list.append(line.strip()) 
    return(ccms_list)

def count_original_peaks(ccms_list, lib):
    """
    Parameters:
        ccms_list (list) : list of strings with ccms ids for the corresponing
        library spectra matrix

        lib (list) : the binned (n,2000) matrix representing the library
        spectra

    The purpose of this function is to count the number of nonzero peaks
    that occur in libary spectra. Saves in ccms_list order to a csv file.
    """
    
    peak_count = []
    for item in lib:
        peak_count.append(np.count_nonzero(item))
        
    df = pd.DataFrame({'peak_count':peak_count})
    df.to_csv("peak_count_1_10000.csv")

def count_added_peaks(lib, noisy, ccms_list=None):
    """
    Parameters:
    
    """
    
    import seaborn as sns
    
    
    added_peaks = []
    for item1, item2 in zip(lib, noisy):
        item1 = np.count_nonzero(item1)
        item2 = np.count_nonzero(item2)
        added_peaks.append(item2 - item1)
    sns.distplot(added_peaks, hist='False')    
    plt.show()
    df = pd.DataFrame({'added_peaks':added_peaks})
    df.to_csv('added_peaks_1_10000.csv')


def percent_of_noise_added_removed(final_lib, denoised_peaks, noisy_peaks):
    added = []
    not_re = []
    re = []
    fpr = []
    tp = []

    for count,(lib, denoi, noi) in enumerate(zip(final_lib, denoised_peaks, noisy_peaks)):
        noise_kept = 0
        noise_added = 0
        noise_removed = 0
        total_real_peaks = 0
        
        false_peak_removal = 0
        true_negatives = 0
        false_negatives = 0
        
       
        for a,b,c in zip(lib, denoi, noi):
            #FP
            #we shouldn't remove it, and we do remove it
            if a > 0 and b == 0:
                false_peak_removal += 1
            #it appeared in the of spectra
            if a > 0:
                total_real_peaks += 1
            #TN
            #we shouldn't remove it, and we don't remove it
            if a > 0 and b > 0:
                true_negatives += 1
            #TP
            #we should remove it, and we do remove it
            if c > 0 and a == 0 and b ==0:
                noise_removed += 1
            
            #FN
            #we should remove it, and we don't remove it
            if c > 0 and a == 0 and b > 0:
                noise_kept += 1
            
            if a == 0 and c > 0:
                noise_added += 1
            
        if total_real_peaks > 0 and noise_added != 0 and true_negatives + false_peak_removal >0 and noise_removed+noise_kept > 0:
            fpr.append(false_peak_removal/(false_peak_removal+true_negatives))
            re.append(noise_removed/(noise_removed+noise_kept))
            
            not_re.append(noise_kept /noise_added)
        if count % 1000 == 0:
            print(count)
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
    import seaborn as sns

    #ax.hexbin(fpr, re,bins='log', gridsize=50, cmap='Purples', mincnt=1)
    #ns.set_style("whitegrid")
    fpr = np.array(fpr)
    re = np.array(re)
    print(len(fpr))
    print(len(re))
    sns.histplot(x=np.array(fpr), y=np.array(re), bins=25, stat='density')
    #sns.scatterplot(x=fpr[:1000], y =re[:1000])
    #ax.plot([0, 1], [0, 1], color = 'black')
    plt.xlabel('False Positive')
    plt.ylabel('True Positive')
    
    """
    sns.distplot(fpr, hist=False, color = 'blue')
    plt.show()
    plt.close()

    ax = sns.distplot(x=re, color = 'blue', hist = False)
    #ns.distplot(x=not_re, color= 'darkgreen', hist=False)
    fig.legend(labels=['Percent of Noise Removed'])
    plt.xlabel('Percent of Noise Removed')
    """
    plt.show()

def filter_data(data, ccms_list):
    new_data = []
    for item in data:
        if item['spectrum_id'] in ccms_list:
            new_data.append(item)
        
    return(new_data)

def main():
    """
    Options & Outputs:
        make_hdf5_lib (bool) : if true, this will create and write a new hdf5 file
        containing binned, normalized, and lowest-level noise removed data from all
        library spectra meeting the search criteria defined in function extract_library_spectra
        along with a text file indicating the order of the ccms ids in the hdf5

        need_full_data (bool) : if true, will fetch all information on all 
        library spectra in a list of dictionaries from GNPS

        load_analyze_msreduce_data (bool) : if true, this will attempt to load
        noisy, original, and msredcue processed data from 3 hdf5 files, and create figures
        representing 1) the cosine distributions of these data alongside the autoencoder data
        2) boxplots showing the improvment in cosine score for both these data and autoencoder
        data (seperate plots) and 3) distibutions showing the number of peaks in the original
        spectra should msreduce have failed (further documented in spoof_noise_recovery.py
        

    Function has several possible use cases, denoted by the boolean expressions.
    Generally, this code serves three main purposes.
    1) To generate mgfs for library search for spiked in noise
    2) To evaluate the results via cosine visualization / mirror plots of library spectra
    with spiked in noise
    3) To visualize autoencoder versus msreduce results

    These dual purposes mean that the files for evaulation, will vary.
    """

    library_spiked_noise_analysis = False
    msreduce_analysis = False

    make_hdf5_lib = True 
    need_full_data = False
    prepare_denoised_mgf = False

    load_analyze_msreduce_data = False
    
    make_cosine_plot = False
    make_mirror_plot = False
    load_lib_from_hdf5 = False
    
    create_mgfs_lib_search = False
    mirror_plot_top = 0
    mirror_plot_bottom = 0
    filter_data = False

    #library spikd noise analysis files
    denoised_data_filename = './lib_specs/predictions_spoof_master_1_10000.npy'
    noisy_data_filename = './lib_specs/noisy_peak_spoof_master_1_10000.npy'

    name = 'lib_spectra.hdf5'
    num_used_spectra = 10000
    
    if make_hdf5_lib:
        make_hdf5_file(name)
        
        #this that were identified in training data in lib search
        ccms_exclusion = pd.read_csv('./lib_search/ccms_unique_lib_search_training.csv')
        ccms_exclusion_list = ccms_exclusion['SpectrumID'].tolist()
        data = download_library_spectra()
        write_to_hdf5(data, name, ccms_exclusion)
    
    if need_full_data:
        data = download_library_spectra()

    if library_spiked_noise_analysis:
        #read the library data into memory
        lib_data = read_hdf5(name, num_used_spectra) 
        lib_data = bin_reduce_normalize(lib_data)
        final_lib = lib_data

        for i in range(1,20):
            final_lib = np.concatenate((final_lib, lib_data))
        
        ccms_list = read_ccms('ccms_list.txt')[:num_spectra_used]
        og_ccms = ccms_list
        ccms_list = ccms_list * 20

        #load and normalize all autoencoder related data
        noisy_peaks = np.load(noisy_data_filename)
        noisy_peaks = bin_reduce_normalize(noisy_peaks)
        denoised_data = np.load(denoised_data_filename)
    
    if prepare_denoised_mgf:
        mset = set(range(0,2000))
        first_pass = True
        for count, (n, d) in enumerate(zip(noisy_peaks, denoised_data)):
            a = np.nonzero(n)[0].tolist()
            nindex = set(a)
            oindex = mset - nindex
            zeroi = list(oindex)
            d[zeroi] = 0.0
            denoised_data[count] = d

        #in case we'd like to save this for future use        
        #np.save('./denoised_processed_data_1_10000.npy', denoised_data)
    
    if msreduce_analysis: 
        msr = np.load('./msr_final.npy')
        noisy = np.load('./noise_added.npy')
        og_msr = np.load('./original.npy')
        ccms_list = read_ccms('new_ccms_list.txt')

    #fetch_molecular_formula(ccms_list, data)
    #fetch_compound_name(ccms_list, data)
    #count_original_peaks(ccms_list, final_lib)
    #count_added_peaks(final_lib, noisy_peaks, ccms_list)      
    #percent_of_noise_added_removed(final_lib, denoised_data, noisy_peaks)
    
    if make_mirror_plot:
        mirror_plots(noisy_peaks[mirror_plot_top], final_lib[mirror_plot_bottom].T)   
        mirror_plots(denoised_data[mirror_plot_top], final_lib[mirror_plot_bottom].T)
   
    
    if make_cosine_plot and not load_analyze_msreduce_data:
        cosine_score(denoised_data, final_lib, noisy_peaks)
    if make_cosine_plot and load_analyze_msreduce_data: 
        cosine_score(denoised_data, final_lib, noisy_peaks, noisy, msr, og_msr)
     
    if create_mgfs_lib_search:
        #this cuts down on the amount of data iterated when making mgf 
        if filter_data:
            data = filter_data(data, og_ccms)
            json_string = json.dumps(data)
            with open('filtered_data.json','r') as json_file:
                json.dumps(json_string, json_file)
        else:
            with open('filtered_data.json', 'r') as json_file:
                data = json.load(json_file)
            data = ast.literal_eval(data)

        #takes the original data, the data and the spectra id number and formats an mgf for library search
        reformat_mgf(final_lib, data, ccms_list, 'lib_data_42.5')
        reformat_mgf(denoised_data, data, ccms_list, 'denoised_42.5')
        reformat_mgf(noisy_peaks, data, ccms_list, 'noisy_42.5')
    

if __name__ == "__main__":
    main()

