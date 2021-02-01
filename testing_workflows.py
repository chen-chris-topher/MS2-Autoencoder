import math
import sys
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py


def failed_peak_analysis(fail_low, fail_high, fail_pre, nof_low=None, nof_high=None,nof_pre=None):
    from mpl_toolkits import mplot3d
    low_peak = []
    high_peak = []
    pre_peak = []

    nof_low_peak = []
    nof_high_peak =[]
    nof_pre_peak = []

    inten_fail_low =[]
    inten_fail_high = []
    inten_fail_pre = []

    inten_low = []
    inten_high = []
    inten_pre = []
    
    if nof_low != None:
        for low, high, pre in zip(nof_low, nof_high, nof_pre):
            nof_low_peak.append(np.count_nonzero(low))
            nof_high_peak.append(1-cosine(high,pre))
            nof_pre_peak.append(np.count_nonzero(high))
            
            inten_fail_low.append(np.mean(low))
            inten_fail_high.append(np.mean(high))
            inten_fail_pre.append(np.mean(pre))


    for low, high, pre in zip(fail_low, fail_high, fail_pre):
        low_peak.append(np.count_nonzero(low))
        high_peak.append(1-cosine(high,pre))
        pre_peak.append(np.count_nonzero(high))
    
        inten_low.append(np.mean(low))
        inten_high.append(np.mean(high))
        inten_pre.append(np.mean(pre))


    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    
    #ax.scatter3D(nof_high_peak, nof_low_peak, nof_pre_peak, color = "green")
    ax.scatter3D(high_peak, low_peak, pre_peak, color = 'red')
    ax.set_xlabel('Cos Score High-Pre', fontweight ='bold') 
    ax.set_ylabel('Low Peak Count', fontweight ='bold') 
    ax.set_zlabel('High Peak Count', fontweight ='bold')
    plt.show()


def plot_cos_dist(low_spectra, high_spectra, predicted_spectra):
    import seaborn as sns
    from mpl_toolkits import mplot3d
    original_cos = []
    new_cos = []
    low_cos = []
    
    num_peaks_high = []
    num_peaks_low = []
    num_peaks_predicted = []

    fail_low = []
    fail_high = []
    fail_pre = []

    no_fail_low = []
    no_fail_high = []
    no_fail_pre = []

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
        
        num_peaks_high.append(np.count_nonzero(high))
        num_peaks_low.append(np.count_nonzero(low))
        num_peaks_predicted.append(np.count_nonzero(pre))
        

        """ 
        if  1-cosine(low,pre) - (1-cosine(high, pre)) > 0.2: 
            fail_low.append(low)
            fail_high.append(high)
            fail_pre.append(pre)
            #mirror_plot(low, high)
            #mirror_plot(low, pre)
          
        else:
            no_fail_low.append(low)
            no_fail_high.append(high)
            no_fail_pre.append(pre)
        """

        """      
        if 1-cosine(low,pre)  - (1-cosine(high, pre)) >= 0.2 and 1-cosine(low,high) > 0.5:
            fail_low.append(low)
            fail_high.append(high)
            fail_pre.append(pre)
            mirror_plot(low,high)
            mirror_plot(pre, high)
        """ 
            
     
    #ailed_peak_analysis(fail_low, fail_high, fail_pre)
    #failed_peak_analysis(fail_low, fail_high, fail_pre, no_fail_low, no_fail_high, no_fail_pre) 
    #sys.exit(0)
     
    print( "Low High Cos Avg", sum(original_cos)/ len(original_cos))
    print("High Pred Cos Avg", sum(new_cos) / len(new_cos))
    print("Low Pred Cos Avg", sum(low_cos) / len(low_cos))
    
    """ 
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    ax.scatter3D(original_cos, new_cos, low_cos, color = "green")
    ax.set_xlabel('High-Low Cos', fontweight ='bold') 
    ax.set_ylabel('High-Predicted Cos', fontweight ='bold') 
    ax.set_zlabel('Low-Predicted Cos', fontweight ='bold')
    """

    
    #ax = sns.jointplot(original_cos, new_cos, kind = 'hex')
    #ax = sns.scatterplot(new_cos, num_peaks_predicted)
    #ax = sns.scatterplot(original_cos, new_cos)
    #ax = sns.distplot(original_cos, color='purple')
    ax = sns.distplot(new_cos, color='green')
    #ax = sns.distplot(low_cos, color ='blue')
    ax.set(xlabel='Original Cos', ylabel='New Cosine')
    plt.show()

def load_np(filename):
    data = np.load(filename)
    return(data)

def open_hdf5(target):
    hf = h5py.File(target, 'r')
    low_dset = hf.get('low_peaks')[:1152]
    high_dset = hf.get('high_peaks')[:1152]
    return(low_dset, high_dset)


def mirror_plot(spec1, spec2):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
    spec2 = spec2.flatten()
    spec1 = spec1.flatten()
    print(spec1.shape, spec2.shape)
    print(type(spec1[0]), type(spec2[0]))
    spec_list = [spec1, spec2]
    mz = list(range(0,2000))
    
    spectra = []
    
    for thing, m in zip(spec1,mz):
        if thing != 0:
            print(thing, m)

    for spec in spec_list:
        spectra.append(sus.MsmsSpectrum(0, 0,0, mz, spec,retention_time=0))
    
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
        if l > 1.0 or h > 1.0:
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

#determine # peaks on average in high versus low
def average_peak_intensity(low_peaks, high_peaks, predicted_peaks):
    from mpl_toolkits import mplot3d

    low_points = []
    high_points = []
    pre_points = []
    for low, high, pre in zip(low_peaks, high_peaks, predicted_peaks):
        low_points.append(np.sum(low)/np.count_nonzero(low))
        high_points.append(np.sum(high)/np.count_nonzero(high))
        pre_points.append(np.sum(pre)/np.count_nonzero(pre))
    
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
   
    ax.scatter3D(high_points, low_points, pre_points, color = 'blue')
    ax.set_xlabel('High', fontweight ='bold') 
    ax.set_ylabel('Low', fontweight ='bold') 
    ax.set_zlabel('Predicted', fontweight ='bold')
    plt.show()



def main():
    predict_target = './predictions.npy'
    #hdf5_target = 'shuffled_data.hdf5'
    #hdf5_target = './hong_reduced.hdf5'
    hdf5_target = './data_3_noise_filter.hdf5'
    model_target = './models/conv1d/conv1d_34.h5'
    history = './models/conv1d/conv1d_34_history.pickle'


    #low, high = open_hdf5('./shuffled_data.hdf5')
    low_spectra, high_spectra = open_hdf5(hdf5_target)
    predictions_0 = load_np(predict_target)
    #print(predictions.shape)

    
    #normalize_data(low_spectra, high_spectra)
    #low_high_inten_dist(low_spectra, predictions)
    
    
    fp = True 

    for row in predictions_0:
        
        row = row.flatten()     
         
        sum_val = np.sum(row)

        row = row / sum_val
        row[row < 0.05] = 0.0

        if str(row[0]) == 'nan':
            continue
        
        if fp == True:
            fp = False
            predictions = row
        else:
            predictions = np.vstack((predictions, row))
    sys.exit(0)
    #print(predictions)
    low_high_inten_dist(low_spectra, predictions)

    #print(low_spectra)
    #print(high_spectra)
    explore_data_pca(low_spectra, high_spectra, predictions)
    average_peak_intensity(low_spectra, high_spectra, predictions)
    #loss_curve(history)
    plot_cos_dist(low_spectra, high_spectra, predictions)
    #mirror_plot(predictions[0], high_spectra[0])

if __name__ =="__main__":
    main()
