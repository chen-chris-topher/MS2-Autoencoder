import math
import sys
from sklearn.decomposition import PCA
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
    #ax = sns.distplot(low_cos, color ='blue')
    #ax.set(xlabel='Cosine Score Low-High', ylabel='Cosine Score High-Predicted')
    plt.show()

def load_np(filename):
    data = np.load(filename)
    return(data)

def open_hdf5(target):
    hf = h5py.File(target, 'r')
    low_dset = hf.get('low_peaks')[:2000]
    high_dset = hf.get('high_peaks')[:2000]
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

#view pca of data
def explore_data_pca(low, high, optional=None):
    import seaborn as sns
    pca = PCA(n_components = 5)

    low_label = ['low'] * len(low)
    high_label = ['high'] * len(high)
    low_label.extend(high_label)
     
    X = np.concatenate((low, high), axis=0) 
    
    if optional is not None:
        opt_label = ['third party'] * len(optional)
        low_label.extend(opt_label)
        optional = np.squeeze(optional)
        print(X.shape)
        X = np.concatenate((X, optional))
    print(X.shape)
    sklearn_matrix = pca.fit_transform(X)
    df = pd.DataFrame(sklearn_matrix)
    df['label'] = low_label
    
    #select_features_pca(sklearn_matrix, pca)

    sns.scatterplot(x=3, y=4, hue = 'label', data =df)
    plt.show()

#look at the distribution of MS/MS peak intensities
def low_high_inten_dist(low, high):
    import seaborn as sns
    low_inten = []
    high_inten = []

    for l, h in zip(low, high):
        l = l[l != 0]
        h = h[h != 0]
        low_inten.extend(l)
        high_inten.extend(h)

    
    for l, h in zip(low_inten, high_inten):
        if l < 0.05 or h < 0.05:
            print(l, h)
    
    sns.distplot(low_inten, color='green')
    sns.distplot(high_inten)
    plt.show()


def average_low_high_peaks(low, high):
    import seaborn as sns
    low_nonzero = []
    high_nonzero = []

    for l, h in zip(low, high):
        low_nonzero.append(np.count_nonzero(l))
        high_nonzero.append(np.count_nonzero(h))

    sns.distplot(low_nonzero, color='green')
    sns.distplot(high_nonzero)
    plt.show()

#show the distribution of fragments masses across all data
def fragment_mz_dist(low, high):
    import seaborn as sns
    
    low = np.where(low > 0.0, 1, 0)
    high = np.where(high > 0.0, 1, 0)

    low_dist = np.sum(low, axis = 0)
    low_d = []
   
    for count, num in enumerate(low_dist):
        temp_list = [count] * num
        low_d.extend(temp_list)    
            
    high_dist = np.sum(high, axis = 0)
    high_d = []
   
    for count, num in enumerate(high_dist):
        temp_list = [count] * num
        high_d.extend(temp_list)    
    
    
    sns.distplot(low_d)
    sns.distplot(high_d)
    plt.show()    
 
#calculate loadings for pca and print
def select_features_pca(X, pca):
    eigenvalues = pca.explained_variance_
    all_loadings = []
    for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
        loadings = np.dot(eigenvector.T, np.sqrt(eigenvalue))
        all_loadings.append(loadings)

    first_loadings = all_loadings[0]
    print(min(first_loadings))
    print(first_loadings.tolist().index(min(first_loadings)))

#sample code to normalize the data properly
def normalize_data(low, high):
    fp = True
    for l, h in zip(low, high):
        l_max= np.max(l, axis=0)
        h_max = np.max(h, axis=0)

        if l_max > 0 and h_max > 0:
            norm_l = np.true_divide(l, l_max)
            norm_h = np.true_divide(h, h_max)

            norm_l[norm_l < 0.05] = 0
            norm_h[norm_h < 0.05] = 0
            
            if fp is False:
                l_vec = np.vstack((l_vec, norm_l))
                h_vec = np.vstack((h_vec, norm_h))

            else:
                l_vec = norm_l
                h_vec = norm_h
                fp = False
  
    low_high_inten_dist(l_vec, h_vec)

def main():
    predict_target = './predictions.npy'
    #hdf5_target = 'shuffled_data.hdf5'
    hdf5_target = './data_3_noise_filter.hdf5'
    model_target = './models/conv1d/conv1d_33.h5'
    history = './models/conv1d/conv1d_33_history.pickle'


    low, high = open_hdf5('./shuffled_data.hdf5')
    low_spectra, high_spectra = open_hdf5(hdf5_target)
    

    predictions = load_np(predict_target)
    print(predictions.shape)

    explore_data_pca(low_spectra, high_spectra, low)
    #normalize_data(low_spectra, high_spectra)
    low_high_inten_dist(low_spectra, high_spectra)
    
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
    loss_curve(history)
    plot_cos_dist(low_spectra, high_spectra, predictions)
    #mirror_plot(predictions[0], high_spectra[0])

if __name__ =="__main__":
    main()
