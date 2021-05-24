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
    from mpl_toolkits import mplot3d
    original_cos = []
    new_cos = []
    low_cos = []
   
    num_peaks_removed_prediction = []

    high_low_ratio = [] #
    pre_low_ratio = [] #describes the average intensity ratio for predicted / low spectra
    pre_high_ratio = []

    count = 0
    ncount = 0
    lcount = 0


    five = []
    six = []
    seven = []
    eight = []
    nine = []
    label = []
    val = []
    for low, high, pre in zip(low_spectra, high_spectra, predicted_spectra):
        if str(1-cosine(low,high)) != 'nan':
            pass
        else:
            count += 1
            continue
     
        og = 1-cosine(low,high)
        nw = 1-cosine(high, pre)
        
        if og >= 0.5 and og < 0.6:
            five.append(nw - og)
            label.append('0.5-0.6')
        if og >=0.6 and og < 0.7:
            six.append(nw - og)
            label.append('0.6-0.7')
        if og >= 0.7 and og < 0.8:
            seven.append(nw - og)
            label.append('0.7-0.8')
        if og >= 0.80 and og < 0.9:
            eight.append(nw - og)
            label.append('0.8-0.9')
        if og >= 0.9:
            nine.append(nw-og)
            label.append('0.9-0.98')

        val.append(nw-og)
        
        #if nw > og + 0.2:
            #print(count)

        original_cos.append(1-cosine(low,high))
        new_cos.append(1-cosine(high, pre))
        low_cos.append(1-cosine(low,pre))    
        
        num_peaks_high.append(np.count_nonzero(high))
        num_peaks_low.append(np.count_nonzero(low))
        num_peaks_predicted.append(np.count_nonzero(pre))
        count += 1

  
    print(len(five), len(six), len(seven), len(eight), len(nine))
    print(count)
    print("Average High Number of Peaks", np.sum(num_peaks_high)/count)
    print("Average Low Number of Peaks", np.sum(num_peaks_low)/count)
    print("Average Predicted Number of Peaks", np.sum(num_peaks_predicted)/count)

    print( "Low High Cos Avg", sum(original_cos)/ len(original_cos))
    print("High Pred Cos Avg", sum(new_cos) / len(new_cos))
    print("Low Pred Cos Avg", sum(low_cos) / len(low_cos))
    

    """
    sns.set_style("whitegrid")
    ax = sns.scatterplot(original_cos, new_cos, color = 'purple')
    ax.plot([0.5, 1], [0, 1], color = 'orange')
    plt.show()
    plt.close()
    """

     
    fig, ax = plt.subplots(ncols=1, figsize=(8, 8))
    hb = ax.hexbin(original_cos, new_cos, bins = 'log', gridsize=50, cmap='Purples', mincnt=1)
    ax.axis([0.5, 1, 0, 1])
    ax.set_title("Cosine Improvement on Validation Data")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('Density (Log Scale)')
    plt.xlabel('Original Cosine Score')
    plt.ylabel('New Cosine Score')
    ax.plot([0.5, 1], [0, 1], color = 'orange')

    plt.show()    
    plt.close() 
    sns.set_style("whitegrid")
    
    ax = sns.boxplot(x=label, y=val, order=["0.5-0.6", "0.6-0.7", "0.7-0.8","0.8-0.9", "0.9-0.98"], palette="Purples")
    ax.axhline(0.0, color = 'orange', alpha = 5)
    ax.set(xlabel='Cosine Score Low-High', ylabel='Improvement in Cosine Score')
    
    plt.show()
    plt.close()
    
    fig = plt.figure(figsize=(10,6))
    ax = sns.distplot(original_cos, color = 'orange')
    sns.distplot(new_cos, color='purple')
    ax.set(xlabel='Cosine Score')
    ax.set(xlim=(0.5, 1.0))
    fig.legend(labels=['Cosine of Low (Input) vs. High (Target)','Cosine of Predicted vs. High(Target)'])
    plt.show()

def load_np(filename):
    data = np.load(filename)[:300000]
    return(data)

def open_hdf5(target, num_train_specs, upper_bound=None):
    """
    Parameters:
        target (str) : the full path of the hdf5 to be opened

        num_train_specs (int) : the number of spectra used in training

        upper_bound (int) : upper bound of spectra to slice from the hdf5
    
    Returns:
        low_dset (numpy matrix) :
        high_dset (numpy matrix) :

    This function is designed to open an hdf5 file and slice out the appropraite data
    from 'low_peaks' and 'high_peaks' dataset for analysis.

    """

    hf = h5py.File(target, 'r')
  
    #if we dont pass an upper bound, take all spectra not used in training
    if upper_bound is None:
        low_dset = hf.get('low_peaks')[num_train_specs:]
        high_dset = hf.get('high_peaks')[num_train_specs:]
    else:
        low_dset = hf.get('low_peaks')[num_train_specs:upper_bound]
        high_dset = hf.get('high_peaks')[num_train_specs:upper_bound]

    return(low_dset, high_dset)

def mirror_plot(spec1, spec2):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
    spec2 = spec2.flatten()
    spec1 = spec1.flatten()
    print(spec1.shape, spec2.shape)
    
    spec_list = [spec1, spec2]
    mz = list(range(0,2000))
    
    spectra = []
    
    print(1-cosine(spec1, spec2))
    for spec in spec_list:
        spectra.append(sus.MsmsSpectrum(0, 0,0, mz, spec,retention_time=0))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()

def acc_plot(history):
    import seaborn as sns
    data = np.load(history, allow_pickle=True)
    df = pd.DataFrame(data)
    test = df['test_acc']
    train = df['train_acc']

    x = range(0,len(test))
    sns.lineplot(x=x,y=test, color ='orange')
    sns.lineplot(x=x,y=train, color='blue')
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
 
#sample code to normalize the data properly
def normalize_data(low, high):
    fp = True
    for count,(l, h) in enumerate(zip(low, high)):
        l_max= np.max(l, axis=0)
        h_max = np.max(h, axis=0)
         
        if l_max > 0 and h_max > 0:
            norm_l = np.true_divide(l, l_max)
            norm_h = np.true_divide(h, h_max)
            
            norm_l[norm_l < 0.05] = 0
            norm_h[norm_h < 0.05] = 0
           
            
            low[count] = norm_l
            high[count] = norm_h
        else:
            print(count)
    return(low, high)
    #low_high_inten_dist(l_vec, h_vec)

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

def model_summary(model_target):
    from tensorflow.keras.models import load_model
    model = load_model(model_target)
    print(model.summary())
   
    #for layer in model.layers: print(layer.get_config(), layer.get_weights())
    
def main():
    """
    Primary use for post-training / predictions data visualization.
    Paramters:
        predict_target (str) : the numpy predcitions file path

        hdf5_target (str) : path to the hdf5 file containing the 
        data on which a prediction was made

        model_target (str) : path to the model being analyzed

        history (str) : path to the .pickle file containing loss
        and accuracy information for the model

        num_test_specs (int) : the number of spectra used for training/testing

    """

    predict_target = './predictions.npy'
    hdf5_target = './shuffle_subset_8.hdf5' 
    model_target = './models/conv1d/conv1d_42.h5'
    history = './models/conv1d/conv1d_42_history.pickle'
    num_test_specs = 3300000

    #prints model summary and weights, must be done in conda environment
    #model_summary(model_target)
     
    low_spectra, high_spectra = open_hdf5(hdf5_target)
    low_spectra, high_spectra= normalize_data(low_spectra, high_spectra)
    predictions = load_np(predict_target)
    
    first_pass = True
    
    for count,predict in enumerate(predictions):
        predict = np.array(predict)
        predict = np.squeeze(predict)
        h_max = np.max(predict, axis=0) 
        norm_h = np.true_divide(predict, h_max)
        norm_h[norm_h < 0.05] = 0
        if count == 0:
            continue
        norm_h = np.expand_dims(norm_h, axis=1)
        predictions[count] = norm_h
    
    mset = set(range(0,2000))
    for count, (n, d) in enumerate(zip(low_spectra, predictions)):
        a = np.nonzero(n)[0].tolist()
        nindex = set(a)
        oindex = mset - nindex
        zeroi = list(oindex)
        d[zeroi] = 0.0 
        predictions[count] = d
    
    
    print("predictions ", predictions.shape)
    print("low ", low_spectra.shape)
    print("high ", high_spectra.shape)
    #low_high_inten_dist(low_spectra, predictions)

    #explore_data_pca(low_spectra, high_spectra, predictions)
    #average_peak_intensity(low_spectra, high_spectra, predictions)
    #loss_curve(history)
    #acc_plot(history)
    plot_cos_dist(low_spectra, high_spectra, predictions)

    spec_num = 0 
    #mirror_plot(low_spectra[spec_num], high_spectra[spec_num])

if __name__ =="__main__":
    main()
