"""

code takes sample datasets, spoofs in low level
random noise, then tries to denoise. calculates 
number of peaks successfully recovered and peaks
missed in the denoising process

also contains functions for mirror plotting
and cosine distribution visualization which are
called by other scripts

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
from sklearn.preprocessing import normalize

#set seed for reproducibility
random.seed(10)

#takes two spectra and produces mirror plots
def mirror_plots(spectra_1, spectra_2, troubleshoot=False):
    """
    Attributes
    ----------
    spectra_1 : array-like, dimension (,2000)
        the top spectrum on the mirror plot, masses binned 0-2000

    spectra_2: array-like, dimension (,2000)
        the bottom spectrum on the mirror plot, masses binned 0-2000
    
    troubleshoot: bool
        whether or not to print the nonzero values of both spectra
        to determine if they are what we intend to visualize
    """
    
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus

    mz = np.arange(0,2000)
    spectra = []
    
    #dealing with potential formatting issues
    spectra_1 = np.array(spectra_1)
    spectra_2 = np.array(spectra_2)
    spectra_1 = np.squeeze(spectra_1)
    spectra_2 = np.squeeze(spectra_2)

    #used for trouble-shooting, making sure we are looking at the correct spetra
    if troubleshoot:
        for count,m1 in enumerate(spectra_1):
            if m1 > 0.0:
                print(count)
        print('\n')
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

#the visualization of cosine distributions and boxplots for predicted, real, and noisy data
def cosine_score(peaks_1, peaks_2, peaks_3, peaks_4=None, peaks_5 =None, peaks_6 =None):
    """
    Attributes
    ---------
    peaks_1 : array-like, dimension (n, 2000)
        matrix representing predictions, masses binned 0-2000
    
    peaks_2 : array-like, dimension (n, 2000)
        matrix representing original/ground truth data, masses binned 0-2000

    peaks_3: array-like, dimension (n, 2000)
        matrix representing noisy data, masses binned 0-2000
    
    peaks_4 : array-like, dimension (n, 2000), optional
        matrix representing predictions, masses binned 0-2000
        if passed will plot additional distributions, intended
        for plotting msreduce data alongside autoencoder data

    peaks_5 : array-like, dimension (n, 2000), optional
        matrix representing original/ground truth data, masses binned 0-2000
        if passed will plot additional distributions, intended
        for plotting msreduce data alongside autoencoder data

    peaks_6: array-like, dimension (n, 2000), optional
        matrix representing noisy data, masses binned 0-2000
        if passed will plot additional distributions, intended
        for plotting msreduce data alongside autoencoder data

    """
        
    import seaborn as sns
    #record cosine scores between prediction/high quality data
    all_cos_scores = []

    #record cosine scores between low/high quality data
    side_cos = []
    
    #used for categorizing data to make boxplots
    label = []
    #value associated with category in boxplots
    val = []
   
    #if we have the additional data, we will plot additional visualizations
    if peaks_4 is not None and peaks_5 is not None and peaks_6 is not None:
        if peaks_4.shape != peaks_5.shape != peaks_6.shape:
            print(peaks_4.shape)
            print(peaks_5.shape)
            print(peaks_6.shape)
        
        #store cosine between msreduce processed data and original data
        msr_cos = []
        #store cosine between original and noisy data
        og_noisy = []
        
        #count number of peaks in original data if we failed to process via msreduce
        og_failed_peaks = []
        #count number of peaks in original data if we processed via msreduce
        og_success_peaks = []

        for count,(a,b,c) in enumerate(zip(peaks_4, peaks_5, peaks_6)): 
            #this executes if msreduce failed to process the sample (removed all peaks)
            if np.max(b) == 0:
                og_failed_peaks.append(np.count_nonzero(c))
                                
            else:
                og_success_peaks.append(np.count_nonzero(c))
                msr_cos.append(1-cosine(c,b)) #we can only calculate cosine if a vector has nonzero values

            og_noisy.append(1-cosine(a,c))
        
        """
        Figure Description
        ------------------
        Figure is a distribution plot designed to analyze the origin of msreduce failure.
        When msreduce succeeds, we count the number of peaks in the original spectra, and plot
        is as a distribution, when it fails (removes all peaks). This is meant to test the 
        hypothesis that msreduce over-desnoised spectra, so those starting with lower numbers
        of peaks will have ALL peaks removed.

        Proven correct.
        """
        fig = plt.figure(figsize=(10,6))
        sns.set_style("whitegrid")
        sns.distplot(og_failed_peaks, color='forestgreen', hist=False)
        sns.distplot(og_success_peaks, color='deepskyblue', hist=False)
        ax2 = plt.axes() 
        ax2.set_ylabel('Distribution')
        ax2.set_xlabel('Number of Peaks')
        ax2.set_title('Distribution of Number of Peaks by MSREDUCE Occurred/Failured')
        fig.legend(labels=['Number of Peaks MSREDUCE Failed','Number of Peaks MSREDUCE Occurred'])
        plt.show()
        plt.close()
        
    for count, (peak1, peak2, peak3) in enumerate(zip(peaks_1, peaks_2, peaks_3)):
        #cosine score between the "low" quality and "high" quality data
        low_high_cos = 1 - cosine(peak2, peak3)
        #cosine score between predictions and "high" quality data
        pre_high_cos = 1 - cosine(peak1, peak2)
        
        all_cos_scores.append(pre_high_cos)
        side_cos.append(low_high_cos)
        
        #difference in labeling for boxplots
        """
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
        else:
            continue
        """
        if low_high_cos >= 0.80 and low_high_cos < 0.85:
            label.append('0.80-0.85')
        elif low_high_cos >=0.85 and low_high_cos < 0.90:
            label.append('0.85-0.90')
        elif low_high_cos >= 0.90 and low_high_cos < 0.95:
            label.append('0.90-0.95')
        elif low_high_cos >= 0.95:
            label.append('0.95-1.0')
        else:
            continue
        
        val.append(pre_high_cos-low_high_cos)
    
    """
    Figure Description
    ------------------

    """
    ax = sns.boxplot(x=label, y=val, order=["0.80-0.85", "0.85-0.90","0.90-0.95", "0.95-1.0"], palette="Purples")
    plt.axhline(y=0, color='orange') #horizontal line to mark zero
    #ax = sns.boxplot(x=label, y=val, order=["0.5-0.6", "0.6-0.7", "0.7-0.8","0.8-0.9", "0.9-0.98"], palette="Purples")
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    
    sns.distplot(all_cos_scores, color = 'orange', hist=False)
    sns.distplot(side_cos, color = 'purple', hist = False)
    
    if peaks_4 is not None:
        sns.distplot(msr_cos, color ='blue', hist=False, norm_hist=True)
        #sns.distplot(og_noisy, color= 'darkgreen', hist=False)
    
    ax2 = plt.axes()
    
    ax2.set_ylabel('Distribution')
    ax2.set_xlabel('Cosine Score')
    ax2.set_title('Recovery of Spetcra after Adding Noise')
    #ax2.set_xlim(0.5, 1)
    #ax2.set_ylim(0,1)
    
    if peaks_4 is not None:
        fig.legend(labels=['Predicted Autoencoder vs. Original Spectra','Noisy Autoencoder vs. Original Spectra', 'Predicted MSREDUCE vs. Original Spectra'])
    else:
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
        if i % 1000 == 0:
            print(i)
    return(np.squeeze(np.array(all_predictions)))


#spoofs in low-level noise peaks
def add_noise(peaks, amount): 
    rvs = stats.norm(loc=0.10, scale=0.05).rvs
    noisy_data = np.zeros((peaks.shape[0], peaks.shape[1]), dtype=float)
    print(noisy_data.shape)
    for count, peak in enumerate(peaks): 
        noise = sp.random(1, 1000, density = amount, data_rvs=rvs)
        pad = np.zeros((1, 1000))
        noise = sp.hstack((noise,pad)).toarray()
        noisy_data[count] = peak + noise
    
    print(noisy_data.shape)
    return(noisy_data)

#if the data needs to be collpased and normalized makes 2000 bins
#normalizes to 1 and filters out signals below 0.05
def bin_reduce_normalize(peaks, reduce=False):
    
    
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
    return(peaks)

#load and retrun hdf5 dataset, part or whole
def load_data(dataset_name, num_used_spectra=None):
    hf = h5py.File(dataset_name, 'r')
    if num_used_spectra != None:
        high_dset = hf.get('lib_spectra').value[:num_used_spectra]
    else:
        high_dset = hf.get('lib_spectra').value
    return(high_dset)

def main():
    #the number of library spectra to use for simulation
    num_used_spectra = 10000
    
    #name of where library spectra are stored/binned
    dataset_1 = 'lib_spectra.hdf5'

    #name of the model to use to denoise
    model_name = '../models/conv1d/conv1d_42.h5'
    
    
    #load the hdf5 dataset to test on
    high_peaks = load_data(dataset_1) 
      
    #reduce peaks and normalize
    high_peaks = bin_reduce_normalize(high_peaks)
    print(high_peaks.shape)


    """
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
        noisy_data = normalize(noisy_data, norm='l2')
      
         
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
    """
    
    noisy = np.load('noise_added.npy')
    noisy = normalize(noisy, norm='l2')
    predicted_peaks = denoise(noisy)
    #predicted_peaks = bin_reduce_normalize(predicted_peaks)
    #save the prediction for analysis
    np.save('./predictions.npy', predicted_peaks)
    #np.save('./noisy_peak_spoof_master_1_10000.npy', final_noise)

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


