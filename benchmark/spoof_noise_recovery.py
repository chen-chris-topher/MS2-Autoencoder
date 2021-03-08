"""
code takes sample datasets, spoof in low level
random noise, then tries to denoise. calculates 
number of peaks successfully recovered and peaks
missed in the denoising process
"""

import os 
import sys
import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.stats as stats
from scipy.spatial.distance import cosine

#set seed for reproducibility
random.seed(10)

#model being used for test
model_name = '../models/conv1d/conv1d_36.h5'

#datasets being tested against
DATASET_1 = '../nominal_1000_3000.hdf5'
DATASET_2 = ''
DATASET_3 = ''


#define a custom noise distribution
def distribution(peaks):
    import seaborn as sns
    
    mz = np.arange(0,2000).tolist()
    total = [0.0] * 2000

    for item in peaks:
        item[item > 0.0] = 1
        total = [x + y for x,y in zip(total, item)]
    print(total)
    #total_max = np.max(total)
    #total = np.true_divide(total, peaks.shape[0])
    fig, ax = plt.subplots(figsize=(12, 6))
    ax = sns.scatterplot(mz, total)
    plt.show()

#takes two spectra and produces mirror plots
def mirror_plots(spectra_1, spectra_2):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus

    mz = np.arange(0,2000)
    spectra = []

    spectra_list = [spectra_1, spectra_2]
    for spec in spectra_list:
        spectra.append(sus.MsmsSpectrum(identifier=0, precursor_mz=0, precursor_charge=0, mz=mz, intensity=spec))

    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()

#calculate the cosine score of reocovered data versus original
#pre, actual, noisy
def cosine_score(peaks_1, peaks_2, peaks_3):
    import seaborn as sns
    
    all_cos_scores = []
    side_cos = []

    for count, (peak1, peak2, peak3) in enumerate(zip(peaks_1, peaks_2, peaks_3)):
        low_high_cos = 1 - cosine(peak3, peak2)
        pre_high_cos = 1 - cosine(peak1, peak2)
        


        all_cos_scores.append(pre_high_cos)
        side_cos.append(low_high_cos)

        if low_high_cos > pre_high_cos:
            print(count)
        """
        if (1-cosine(peak1,peak2)) < 0.5:
            print(count)
        """
    # = sns.distplot(all_cos_scores, color = 'orange')
    x = sns.distplot(side_cos, color = 'purple')
    plt.show()

#loads model and predicts on the test dataset
def denoise(peaks):
    from tensorflow.keras import models

    model = models.load_model(model_name)
    all_predictions = []
    batch_size = 1
    i = 0

    while i < peaks.shape[0]:
        test_data = peaks[i:i+batch_size]
        prediction = model.predict(test_data, batch_size=batch_size)
        i += batch_size
        all_predictions.append(prediction)

    return(np.squeeze(np.array(all_predictions)))


#spoofs in low-level noise peaks
def add_noise(peaks):
    first_pass = True
    
    rvs = stats.norm(loc=0.2, scale=0.05).rvs 

    noise = sp.random(peaks.shape[0], peaks.shape[1], density = 0.01, data_rvs=rvs)
    noisy_data = peaks + noise

    """
    #adding noise to each spectra in the dataset
    for count,peak in enumerate(peaks):
        #define some noise parameters where param 1 is mean and param 2 is stdev
        noise = sp.random(peaks.shape[1], peaks.shape[0], density = 0.01) 
        new_signal = peak + noise
        
        
        if first_pass is False:
            noisy_data = np.vstack((noisy_data, new_signal))
        if first_pass is True:
            noisy_data = new_signal
            first_pass = False
        #print(noisy_data.shape)
    """
    return(noisy_data)

#if the data needs to be collpased and normalized makes 2000 bins
#normalizes to 1 and filters out signals below 0.05
def bin_reduce_normalize(peaks, reduce=False):
    first_pass = True
    
    for count, item in enumerate(peaks):
        if reduce:
            item = np.add.reduceat(item, np.arange(0, item.shape[0], 100)) 
        
        max_val = np.amax(item, axis=0)
        
        if max_val == 0:
            continue 

        item = np.true_divide(item, max_val)
        item[item < 0.05] = 0.0
           
        if np.count_nonzero(item) == 0:
            continue
        
        if first_pass is False:
            new_peaks = np.vstack((new_peaks,item))

        if first_pass is True:
            new_peaks = item
            first_pass = False
    
    return(new_peaks)

#load and retrun hdf5 dataset
def load_data(dataset_name):
    hf = h5py.File(dataset_name, 'r')
    high_dset = hf.get('high_peaks').value
    return(high_dset)

def main():
    #load the hdf5 dataset to test on
    high_peaks = load_data(DATASET_1) 
  
    #reduce peaks and normalize
    high_peaks = bin_reduce_normalize(high_peaks)

    #distribution(high_peaks)
    
    #add noise into the dataset of choice
    #noisy_data = add_noise(high_peaks)
    #np.save('./noisy_peak_spoof.npy', noisy_data)
    #print(noisy_data.shape)
  
    #attempt to denoise the noisy data
    #predicted_peaks = denoise(noisy_data)
    #print(predicted_peaks.shape)
    
    #save the prediction for analysis
    #np.save('./predictions_spoof.npy', predicted_peaks)
 
    #load already saved data for analysis
    predicted_peaks =  np.load('./predictions_spoof.npy')
    predicted_peaks = bin_reduce_normalize(predicted_peaks) 
    
    
    #load noisy data
    noisy_peaks = np.load('./noisy_peak_spoof.npy')
    #osine_score(predicted_peaks, high_peaks, noisy_peaks)


    #print(noisy_peaks.shape)
    #print(predicted_peaks.shape)
    #print(high_peaks.shape)

    cosine_score(high_peaks, predicted_peaks)
    
    #print(1-cosine(noisy_peaks[0], high_peaks[0]))
    #print(1-cosine(predicted_peaks[0], high_peaks[0]))
    #mirror_plots(predicted_peaks[0], high_peaks[0])


if __name__ == "__main__":
    main()


