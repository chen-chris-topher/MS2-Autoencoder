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
from scipy.spatial.distance import cosine

#set seed for reproducibility
random.seed(10)

#model being used for test
model_name = '../models/conv1d/conv1d_35.h5'

#datasets being tested against
DATASET_1 = '../test_dataset/hong_dataset.hdf5'
DATASET_2 = ''
DATASET_3 = ''

#calculate the cosine score of reocovered data versus original
def cosine_score(peaks_1, peaks_2):
    import seaborn as sns
    
    all_cos_scores = []
    for peak1, peak2 in zip(peaks_1, peaks_2):
        all_cos_scores.append(1 - cosine(peak1, peak2))
    
    x = sns.distplot(all_cos_scores)
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

    #adding noise to each spectra in the dataset
    for count,peak in enumerate(peaks):
        #define some noise parameters where param 1 is mean and param 2 is stdev
        noise = np.random.normal(0.20, .05, peak.shape)
        new_signal = peak + noise
        
        if first_pass is False:
            noisy_data = np.vstack((noisy_data, new_signal))
        if first_pass is True:
            noisy_data = new_signal
            first_pass = False
        #print(noisy_data.shape)
    return(noisy_data)

#if the data needs to be collpased and normalized makes 2000 bins
#normalizes to 1 and filters out signals below 0.05
def bin_reduce_normalize(peaks):
    new_peaks = np.empty((peaks.shape[0],2000))
    for count, item in enumerate(peaks):
        item = np.add.reduceat(item, np.arange(0, item.shape[0], 100), axis=0)
        max = np.max(item)
        item = np.true_divide(item, max)
        item[item < 0.05] = 0.0
        new_peaks[count] = item

    return(new_peaks)

def load_data(dataset_name):
    hf = h5py.File(dataset_name, 'r')
    high_dset = hf.get('high_peaks').value
    return(high_dset)

def main():
    #load the hdf5 dataset to test on
    high_peaks = load_data(DATASET_1) 
    
    #reduce peaks and normalize
    high_peaks = bin_reduce_normalize(high_peaks)

    #add noise into the dataset of choice
    noisy_data = add_noise(high_peaks)
    print(noisy_data.shape)

    #sanity check to make sure noisy data doesn't look like high data
    #cosine_score(noisy_data, high_peaks)
   
    #attempt to denoise the noisy data
    predicted_peaks = denoise(noisy_data)
    print(predicted_peaks.shape)
    #save the prediction for analysis
    np.save('./predictions_spoof.npy', predicted_peaks)

    #load already saved data for analysis
    predicted_peaks =  np.load('./predictions_spoof.npy')
    cosine_score(predicted_peaks, high_peaks)


if __name__ == "__main__":
    main()


