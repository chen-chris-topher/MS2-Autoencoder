import sys
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py

def plot_cos_dist(low_spectra, high_spectra, predicted_spectra):
    import seaborn as sns
    original_cos = []
    new_cos = []
    low_cos = []
    
    lots_low = 0
    lots_high = 0
    
    for low, high, pre in zip(low_spectra, high_spectra, predicted_spectra):
        if str(1-cosine(low,high)) != 'nan':
            pass
        else:
            continue
        original_cos.append(1-cosine(low,high))
        new_cos.append(1-cosine(high, pre))
        low_cos.append(1-cosine(low,pre))    
        
        """    
        if 1-cosine(low,pre) > 0.7 and 1-cosine(high, pre) < 0.7:
            if np.count_nonzero(low) < np.count_nonzero(high):
                lots_high += 1
                if 1-cosine(low,high) > 0.60:
                    print("Low < High")
                    mirror_plot(high, pre)

            else:
                if 1-cosine(low, high) > 0.60:
                    print("Low more High")
                lots_low += 1
        """
     
    print( "Low High Cos Avg", sum(original_cos)/ len(original_cos))
    print("High Pred Cos Avg", sum(new_cos) / len(new_cos))
    print("Low Pred Cos Avg", sum(low_cos) / len(low_cos))
    #ax = sns.jointplot(original_cos, new_cos, kind = 'hex')
    #ax = sns.scatterplot(original_cos, new_cos)
    ax = sns.distplot(original_cos, color='purple')
    ax = sns.distplot(new_cos, color='green')
    ax = sns.distplot(low_cos, color ='blue')
    #ax.set(xlabel='Cosine Score Low-High', ylabel='Cosine Score High-Predicted')
    plt.show()

def load_np(filename):
    data = np.load(filename)
    return(data)

def open_hdf5(target):
    hf = h5py.File(target, 'r')
    low_dset = hf.get('low_peaks').value
    high_dset = hf.get('high_peaks').value
    return(low_dset, high_dset)


def mirror_plot(spec1, spec2):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
    spec2 = spec2.flatten()
    print(spec1.shape, spec2.shape)
    print(type(spec1[0]), type(spec2[0]))
    spec_list = [spec1, spec2]
    mz = list(range(0,2000))
    
    spectra = []
    
    for thing, m in zip(spec1,mz):
        if thing != 0:
            print(thing, m)

    for spec in spec_list:
        spectra.append(sus.MsmsSpectrum(0, 0,
                                        0, mz, spec,retention_time=0))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()

def loss_curve(history):
    import seaborn as sns
    data = np.load(history, allow_pickle=True)
    df = pd.DataFrame(data)
    test = df['test_loss']
    train = df['train_loss']

    x = range(0,len(test))

    sns.lineplot(x=x,y=test, color ='orange')
    sns.lineplot(x=x,y=train, color='blue')
    plt.show()

def explore_data(low, high):
    import seaborn as sns
    

def main():
    predict_target = './predictions.npy'
    hdf5_target = './data_3_noise_filter.hdf5'
    model_target = './models/conv1d/conv1d_31.h5'
    history = './models/conv1d/conv1d_31_history.pickle'

    low_spectra, high_spectra = open_hdf5(hdf5_target)
    predictions = load_np(predict_target)
    
    """
    fp = True 

    for row in predictions_0:
        row = row.flatten()        
        sum_val = np.sum(row)

        row = row / sum_val
        row[row < 0.05] = 0.0
        
        if fp == True:
            fp = False
            predictions = row
        else:
            predictions = np.vstack((predictions, row))
    """
    #loss_curve(history)
    plot_cos_dist(low_spectra, high_spectra, predictions)
    #mirror_plot(predictions[39], high_spectra[39])

if __name__ =="__main__":
    main()
