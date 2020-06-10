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
    for spectrum in prediction_data:
        pass
        
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
            

def main():
    #parser = argparse.ArgumentParser(description='Parse hdf5 files to mgf.')
    #parser.add_argument('input_hdf5')
    #parser.add_argument('target_dset')

    #input_file = args.input_hdf5
    #target_dset = args.target_df

    reading_binned_json()

    #npy_read('predictions.npy')
    #pickle_read('./models_new/autoencoder/autoencoderhistory.pickle')


if __name__ == "__main__":
    main()
