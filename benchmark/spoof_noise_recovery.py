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
model_name = '../models/conv1d/conv1d_40.h5'

#datasets being tested against
DATASET_1 = 'lib_spectra_3.hdf5'
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
    
    print(spectra_1.shape)
    print(spectra_2.shape)
    
    spectra_1 = np.array(spectra_1)
    spectra_2 = np.array(spectra_2)

    spectra_1 = np.squeeze(spectra_1)
    spectra_2 = np.squeeze(spectra_2)

    for count,m1 in enumerate(spectra_1):
        if m1 > 0.0:
            print(count)

    print("MS2")
    for count,m1 in enumerate(spectra_2):
        if m1 > 0.0:
            print(count)




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
    label = []
    val = []
    for count, (peak1, peak2, peak3) in enumerate(zip(peaks_1, peaks_2, peaks_3)):
        low_high_cos = 1 - cosine(peak3, peak2)
        pre_high_cos = 1 - cosine(peak1, peak2)
        
        all_cos_scores.append(pre_high_cos)
        side_cos.append(low_high_cos)

        if low_high_cos >= 0.5 and low_high_cos < 0.6:
            label.append('0.5-0.6')
        elif low_high_cos >=0.6 and low_high_cos < 0.7:
            label.append('0.6-0.7')
        elif low_high_cos >= 0.7 and low_high_cos < 0.8:
            label.append('0.7-0.8')
        elif low_high_cos >= 0.80 and low_high_cos < 0.9:
            label.append('0.8-0.9')
        elif low_high_cos >= 0.9:
            label.append('0.9-0.98')
        val.append(pre_high_cos-low_high_cos)
    ax = sns.boxplot(x=label, y=val, order=["0.5-0.6", "0.6-0.7", "0.7-0.8","0.8-0.9", "0.9-0.98"], palette="Purples")
    plt.show()
    plt.clf()

    fig = plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    sns.distplot(all_cos_scores, color = 'orange')
    sns.distplot(side_cos, color = 'purple')
    ax2 = plt.axes()
    ax2.set_ylabel('Distribution')
    ax2.set_xlabel('Cosine Score')
    ax2.set_title('Recovery of Spetcra after Adding Noise')
    fig.legend(labels=['Predicted vs. Original','Noisy vs. Original'])
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
def add_noise(peaks, amount):
    first_pass = True
    
    #loc = mean scal = std dev
    rvs = stats.norm(loc=0.10, scale=0.05).rvs 
    noise = sp.random(peaks.shape[0], 1000, density = amount, data_rvs=rvs)
    pad = np.zeros((peaks.shape[0], 1000))
    noise = sp.hstack((noise,pad))
   
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
        item = np.array(item)
        item = np.squeeze(item)
        
        if reduce:
            item = np.add.reduceat(item, np.arange(0, item.shape[0], 100)) 
        
        max_val = np.amax(item, axis=0)
             
        if max_val == 0:
            continue 

        
        item = np.true_divide(item, max_val)
        
        item[item < 0.05] = 0.0
           
        if np.count_nonzero(item) == 0:
            continue
        peaks[count] = item
        """
        if first_pass is False:
            new_peaks = np.vstack((new_peaks,item))

        if first_pass is True:
            new_peaks = item
            first_pass = False
        """
    return(peaks)

#load and retrun hdf5 dataset
def load_data(dataset_name):
    hf = h5py.File(dataset_name, 'r')
    high_dset = hf.get('lib_spectra').value[:2000]
    return(high_dset)

def main():
    #load the hdf5 dataset to test on
    high_peaks = load_data(DATASET_1) 
  
    #reduce peaks and normalize
    high_peaks = bin_reduce_normalize(high_peaks)

    #distribution(high_peaks)
    first_pass = True
    #add noise into the dataset of choice
    range_val = range(1,21)
    for amount in range_val:
        print(amount/1000)
        amount = amount / 1000
        noisy_data = add_noise(high_peaks, amount)
         
        from sklearn.preprocessing import normalize
        
        #we need to re-norm to l2 prior to using model
        for i in range(0, len(noisy_data)): 
            noisy_data[i] = normalize(noisy_data[i], norm='l2')
      
         
        #attempt to denoise the noisy data
        predicted_peaks = denoise(noisy_data)
        predicted_peaks = bin_reduce_normalize(predicted_peaks)    
        
        #predicted_peaks = bin_reduce_normalize(predicted_peaks)
        noisy_data = bin_reduce_normalize(noisy_data)
    
        if first_pass is False:
            final_noise = np.concatenate((final_noise, noisy_data))
            final_predicted = np.concatenate((final_predicted, predicted_peaks))
  
        else:
            final_noise = noisy_data
            final_predicted = predicted_peaks
            first_pass = False


    #save the prediction for analysis
    np.save('./predictions_spoof_master_2.npy', final_predicted)
    np.save('./noisy_peak_spoof_master_2.npy', final_noise)

    #load already saved data for analysis
    #predicted_peaks =  np.load('./predictions_spoof.npy')
    #predicted_peaks = bin_reduce_normalize(predicted_peaks) 
    
    
    #load noisy data
    #noisy_peaks = np.load('./noisy_peak_spoof.npy')
    #cosine_score(predicted_peaks, high_peaks, noisy_peaks)


    #print(noisy_peaks.shape)
    #print(predicted_peaks.shape)
    #print(high_peaks.shape)

    #cosine_score(high_peaks, predicted_peaks)
    
    #print(1-cosine(noisy_peaks[0], high_peaks[0]))
    #print(1-cosine(predicted_peaks[0], high_peaks[0]))
    #mirror_plots(np.squeeze(noisy_peaks[6]), predicted_peaks[6])


if __name__ == "__main__":
    main()


