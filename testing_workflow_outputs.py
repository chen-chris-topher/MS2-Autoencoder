import h5py
from scipy.spatial.distance import cosine
import numpy as np
import copy
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pyteomics import mgf

###fuction writes to a mgf file
def write_mgf(spectra_list):
    mgf.write(spectra = spectra_list, output = "./GNPS2.mgf", write_charges = False, use_numpy = True)

###function takes in MS/MS information and formats it into appropriate mgf format
def format_mgf(feat_id_count, precursor_ion, mass_array, inten_array, spectra_list, filepath='None'):
    '''
    feat_id_count
    precursor_ion
    mass_array
    intent_array
    spectra_list
    spoof_quant_table
    '''

    feat_id_count += 1                    
    parameters = {"FEATURE_ID": feat_id_count, 'FILENAME': filepath, 'PEPMASS' : str(precursor_ion), "SCANS": feat_id_count, "RTINSECONDS" : 0.0, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN" : 0.0} 
    spectra = [{'m/z array' : mass_array, 'intensity array': inten_array, 'params': parameters}]                                                            
    
    spectra_list.extend(copy.deepcopy(spectra))
       
    return(spectra_list, feat_id_count)

###function takes npy with the predicted spectra in and outputs relavant information
def npy_read(filename):
    prediction_data =  np.load(filename)
    return(prediction_data)
        
###read autoencoder traning history and build a learning curve
def pickle_read(filename):
    history_dict = pd.read_pickle(filename)
    history_df = pd.DataFrame.from_dict(history_dict) 
    build_learning_curve(history_df)

def build_learning_curve(df):
    fig, ax = plt.subplots()
    ax.plot(df['loss'])
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cosine Loss')
    #plt.savefig('./leanring_curve_low_high.png')
    plt.show()

#getting extra things from the original files
def reading_binned_json():
    feat_count_id = 0
    massArray = np.arange(0,2000,0.1)
    spectra_list = []
    quality = 1 #1 means high quality spectra, 0 means low
    
    #make a list of all files/in path to open
    all_directories = [x[0] for x in os.walk('./output_nf')]
    files_to_open = [os.path.join(x, 'ready_array.npz') for x in all_directories]
    counter = 0
    #oop over all files with info
    for filepath in files_to_open:
        if os.path.isfile(filepath):
            pass
        else:
            print("File Does Not Exist")
            continue

        data_load = np.load(filepath, allow_pickle=True) 
        data = data_load['arr_0']
        print(data.shape)
        for scan_list in data:
            #a lot of these are empty, we want to skip em
            if len(scan_list) > 0:
                #print(scan_list[0][quality])
                counter +=1
                #scan_list_quality = scan_list[0][quality]
                #looping through a scan dict to extract info
                #or key,value in scan_list_quality.items():
                #   final_mass_list = []
                #   final_inten_list = []
                    
                #   if key == 'retentionTime':
                #       precursorRT = value
                #   elif key == 'precursorMz':
                #       precursorMass = value
                #   elif key == 'precursorIntensity':
                #       precursorIntensity = value
                #   elif key == 'intensity array':
                #       intensityArray = value
                #   else: 
                #       print("weird lack of key")
                #getting rid of blank spectra values 
                #or inten, mass in zip(intensityArray, massArray):
                #   if inten != 0:
                #       final_mass_list.append(mass)
                #       final_inten_list.append(inten)
                #writing the scan to a spectra list
                #spectra_list, feat_id_count = format_mgf(feat_count_id, precursorMass, final_mass_list, final_inten_list, spectra_list)
                #feat_id_count += 1
    print(counter)
    print(len(spectra_list))
    write_mgf(spectra_list)

###fuction outputs a distribtuion of cosine scores for predictions versus actual data
def cosine_distributions(prediction, actual):
    import seaborn as sns
    all_cos_scores = []
   

    for pre, act in zip(prediction, actual):
        
        act = np.add.reduceat(act, np.arange(0, len(act), 100))
        cos_score = cosine(act,pre)
        all_cos_scores.append(1 - cos_score)
    print(all_cos_scores)
    x = sns.distplot(all_cos_scores)
    plt.show()

###read in actual spectra vectors from the extracted data
def read_in_actual_spectra(input_file, target_dataset):
    hf = h5py.File(input_file, 'r')
    dset = hf[target_dataset]
    return(dset)


###makes mirror plots of all predicted versus actual spectra pass
def mirror_plots(actual, predicted):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus

    print(actual.shape)
    print(predicted.shape)

    #actual = np.add.reduceat(actual, np.arange(0, len(actual), 100))
#    predicted = np.add.reduceat(predicted, np.arange(0, len(predicted),100))
    
    print(actual.shape)
    print(predicted.shape)
    
    mz = np.arange(0,2000)
    spectra_list = [actual, predicted]
    spectra = []
    for spec in spectra_list:
        spectra.append(sus.MsmsSpectrum(identifier=0, precursor_mz=0, precursor_charge=0, mz=mz, intensity=spec))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    
    #top  = actual, bottom = predicted
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()
    plt.close() 


def scatter_data_validity(predictions, high_spectra, low_spectra):
    x_points = []
    y_points = []
    high_points = []
    low_points = []

        
    
    for pre, high, low in zip(predictions, high_spectra, low_spectra):
        
        high = np.add.reduceat(high, np.arange(0, len(high), 100))
        low = np.add.reduceat(low, np.arange(0, len(low), 100))
        y = cosine(pre,high)
        y = 1 - y
        if str(y) != 'nan':
            y_points.append(y)

            x = cosine(low, high)
            x = 1- x
            x_points.append(x)
            
            high_points.append(high)
            low_points.append(low)
            

    
    plots_from_quadrants(x_points, y_points, low_points, high_points)
    plt.plot(x_points, y_points, 'o', color='black');
    plt.ylabel('cosine predicted vs high')
    plt.xlabel('cosine low vs high')
    plt.show()

def plots_from_quadrants(x_points, y_points, low_points, high_points):
    #print(x_points[:100])
    #print(y_points[:100])
    
    x_threshold = 0.95
    y_threshold = 0.95

    for count,(x,y) in enumerate(zip(x_points, y_points)):
        if x >= x_threshold:
            if y >= y_threshold:
                print(count)
                break


    POI = 181 

    print(x_points[POI], y_points[POI])
    mirror_plots(low_points[POI], high_points[POI])
    return

def main():
    parser = argparse.ArgumentParser(description='Parse hdf5 files to mgf.')
    parser.add_argument('input_hdf5')
    parser.add_argument('target_dset')

    args = parser.parse_args()
    input_file = args.input_hdf5
    target_dset = args.target_dset

    #eading_binned_json()
    low_dset = read_in_actual_spectra(input_file, 'low_peaks')
    high_dset = read_in_actual_spectra(input_file, 'high_peaks')
    predictions = npy_read('predictions.npy')
    print(len(predictions))
    #cosine_distributions(predictions, high_dset)


    #pickle_read('./models/autoencoder/cos_red_autoencoderhistory.pickle')
    scatter_data_validity(predictions, high_dset, low_dset)
   

    #print(predictions[0].shape) 
    #mirror_plots(high_dset[1], predictions[1])
    #tester =  np.add.reduceat(high_dset[1], np.arange(0, len(high_dset[0]), 100))
    #print(1 - cosine(tester[1], predictions[1]))

if __name__ == "__main__":
    main()
