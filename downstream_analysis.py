import base64
import copy
import matplotlib.pyplot as plt
import ms2_model
import math
import h5py
import json
import os
import sys
sys.path.insert(1, './bin')
import extract_mzxml as em
import ms2_model as ms

import numpy as np

import argparse
from scipy.stats import binned_statistic
from tensorflow.keras import models
from pyteomics import mzxml
from pyteomics import mgf
def read_mzXML(filename):
    data = mzxml.MzXML(filename)
    return(data)

def extract_spectra(data, filename):
    print("Begin file extraction")
    directory = './test_dataset'
     
    intensity_array_list = []

    em.count_MS2(data)   
    id_list_ms2 = em.find_MS2(data, directory)
     
    processed_dict = {}
    for index in id_list_ms2:
        scan = int(data[index].get('id'))
        rt = data[index].get('retentionTime')
        intensity = data[index].get('precursorMz')[0].get('precursorIntensity')
        mz = data[index].get('precursorMz')[0].get('precursorMz')
        mz_array = data[index].get('m/z array').tolist()
        og_intensity_array = data[index].get('intensity array').tolist()
        
        binned_intensity, binned_mz, _ = binned_statistic(mz_array, og_intensity_array, statistic='sum', bins=2000, range=(0, 2000))
        intensity_array = binned_intensity.tolist() 
        max = np.max(intensity_array)
        intensity_array = np.true_divide(intensity_array, max)
        intensity_array[intensity_array < 0.05] = 0.0
        
        intensity_array_list.append(intensity_array)
        
        processed_dict[index]  = {'scan':scan, 'retentionTime':rt, \
        'precursorMz':mz, 'precursorIntensity':intensity, 'mz array':mz_array,\
        'intensity array': intensity_array} #intensity array
    
    #rint(processed_dict)
    #son = json.dumps(processed_dict)
    #ith open(directory + '/output.json', 'w') as output:
    #   output.write(json)
    
    return(processed_dict, np.array(intensity_array_list))

#make predictions for every spectra in the file
def predict_high(model_name, intensity_array_list):
    model = models.load_model(model_name)
    all_predictions = []
    batch_size = 1
    i = 0
    #print(intensity_array_list.shape)
    
    while i < intensity_array_list.shape[0]:
        test_data = intensity_array_list[i:i+batch_size]
        prediction = model.predict(test_data, batch_size=batch_size)
        i += batch_size
        
        all_predictions.append(prediction)
    
    return(np.squeeze(np.array(all_predictions)))

#take predictions and replace the typical scans within the file
def replace_scans(processed_dict, prediction_matrix, filename):
    file_data = mzxml.MzXML(filename)
    count = 0
    spectra_list = []
    
    print(prediction_matrix.shape)
    for scan, data in processed_dict.items():
        og_intensity_array = data['intensity array']
        og_mz_array = data['mz array'] 
        new_mz = []
        new_inten = []
        
        """
        #print(og_mz_array)
        track_bins = {}
        
        #looks through the current inten/mass arrays and looks at the predictions to see if the masses appear - if they don't we ignore it
        for ogmz, ogin in zip(og_mz_array, og_intensity_array):
            bin_val = math.floor(ogmz)
            if bin_val >= 2000:
                continue
            
            predict_inten = prediction_matrix[count,:].tolist()
            predict_inten = predict_inten[bin_val]
            
            #if we found values in this bin we ignore it
            if predict_inten != 0:
                if bin_val not in track_bins:
                    track_bins[bin_val] = ogin
                    new_mz.append(ogmz)
                    new_inten.append(ogin)
                else:
                    compare = track_bins[bin_val]
                    if ogin > compare:
                        for m, i in zip(new_mz, new_inten):
                            if math.floor(m) == bin_val:
                                new_mz.remove(m)
                                new_inten.remove(i)

                        new_mz.append(ogmz)
                        new_inten.append(ogin)
            else:
                pass
	    """
        #rint(track_bins) 
        #print(new_mz)
        #print(new_inten)
        #break

        #ile_data[scan]['m/z array'] = new_mz
        #ile_data[scan]['intensity array'] = new_inten
 		
        predict_inten = prediction_matrix[count,:].tolist()

        mz_possible = list(range(0,2000))
        for p, m in zip(predict_inten, mz_possible):
            if p > 0:
                new_inten.append(p)
                new_mz.append(m)

        
        new_scan = int(file_data[scan].get('id'))
        rt = file_data[scan].get('retentionTime')
        intensity = file_data[scan].get('precursorMz')[0].get('precursorIntensity')
        
        new_inten = [item * intensity for item in new_inten]
        mirror_plots(predict_inten, og_intensity_array)
        break
        mz = file_data[scan].get('precursorMz')[0].get('precursorMz')
        
        parameters = {"FEATURE_ID": count, 'FILENAME': filename, 'PEPMASS' : str(mz), "SCANS": new_scan, "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN" : intensity}
        spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
        spectra_list.extend(copy.deepcopy(spectra))
 
        count += 1
   
    mgf.write(spectra = spectra_list, output = "./test.mgf", write_charges = False, use_numpy = True)

def mirror_plots(actual, predicted):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
    print(len(predicted))
    print(len(actual))
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


def get_npz():
    target = './ready_array2.npz'
    data = np.load(target)
    data = data['arr_0']

    split_data = np.split(data, 2, axis=1)
    low_peaks = split_data[0]
    high_peaks = split_data[1]

    low_peaks = normalize_data(low_peaks)
    high_peaks = normalize_data(high_peaks)

    low_peaks = low_peaks.reshape(len(low_peaks), np.prod(low_peaks.shape[1:]))
    high_peaks = high_peaks.reshape(len(high_peaks), np.prod(high_peaks.shape[1:]))

    return(low_peaks, high_peaks)

def normalize_data(peaks):
    new_peaks = np.empty((peaks.shape[0],2000))
     
    for count, item in enumerate(peaks):
        #tem = np.add.reduceat(item, np.arange(0, item.shape[0], 100), axis=0) 
        max = np.max(item) 
        #sys.exit(0)
        item = np.true_divide(item, max)
        item[item < 0.05] = 0.0    
        new_peaks[count] = item

    return(new_peaks)
def supp_info_parse(low_dset, high_dset):
    target = './ready_array.npz'
    data = np.load(target)
    data = data['arr_0']
    
    split_data = np.split(data, 2, axis=1)
    low_supp = split_data[0]
    high_supp = split_data[1]
    
    format_mgf(low_dset, low_supp, 'low')
    format_mgf(high_dset, high_supp, 'high')

def format_mgf(dset, supp, type, features=None):
    count = 0
    spectra_list = []
    
    return_feat = []
    for item, spectra in zip(supp,dset):
        new_inten= []
        new_mz = []
        
        #item = item[0]
   
        rt = item[0]
        intensity = item[1]
        mz = item[2]
        if mz == 0:
            continue
        if intensity == 0:
            continue

        for mass, inten in enumerate(spectra):
            if inten != 0:
                new_inten.append(inten)
                new_mz.append(mass + 0.5)
        new_inten = [item * intensity for item in new_inten]

        return_feat.append((count, mz))
        
        if features != None:
            interest_tuple = features[count]
            low_mz = interest_tuple[1] - 0.5
            high_mz = interest_tuple[1] + 0.5
            if mz <= high_mz and mz >= low_mz:
                parameters = {"FEATURE_ID": count, 'FILENAME': type, 'PEPMASS' : str(mz), "SCANS": count, "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN" : intensity}
                spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
                spectra_list.extend(copy.deepcopy(spectra))
                count += 1
        else:
            parameters = {"FEATURE_ID": count, 'FILENAME': type, 'PEPMASS' : str(mz), "SCANS": count, "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN" : intensity}
            spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
            spectra_list.extend(copy.deepcopy(spectra))   
            count += 1

    mgf.write(spectra = spectra_list, output = "./%s_new.mgf" %type, write_charges = False, use_numpy = True)
    return(return_feat)

def get_hdf5():
    target = './hong_data.hdf5'
    hf = h5py.File(target, 'r')
    low_dset = hf.get('low_peaks').value
    high_dset = hf.get('high_peaks').value
    low_dset = normalize_data(low_dset)
    high_dset = normalize_data(high_dset)
    return(low_dset,high_dset)

def get_sup():
    target = './hong_add_info.hdf5'
    hf = h5py.File(target, 'r')
    low_dset = hf.get('low_peaks').value
    high_dset = hf.get('high_peaks').value
    return(low_dset, high_dset)

def mirror_plot(spec1, spec2):
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus
 
    spec_list = [spec1, spec2]
    mz = list(range(0,2000))
    spectra = []
    
    for thing,m in zip(spec1,mz):
        if thing != 0:
            print(thing, m)
    for spec in spec_list:
        spectra.append(sus.MsmsSpectrum(0, 0,
                                        0, mz, spec,
                                         retention_time=0.0))
 
    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()

def format_mgf_from_file(dset, dset_type, features=None, final_param=None):
    target_file_spectra = './pairs.txt'
    with open(target_file_spectra) as hf:
        data = json.load(hf)
    
    low_supp = np.load('%s.npy' %dset_type)
    
    spectra_list = [] #defines where we will save mgf appropriate spectra
    count = 0
    return_feat = []
    success_count = 0
    mass_range = range(0,2000)

    #loop over dataset and assoicated file names
    for item, spectra in zip(low_supp, dset):     
        if 'nan' == spectra[0]:
            continue
        
        new_inten = []
        new_mz = [] 
        rt = round(float(item[0]), 6)
        intensity = float(item[1])
        mz = round(float(item[2]), 6)
        key = -1 
        

        if mz == 0:
            continue
        if intensity == 0:
            continue
        
        if features:
            a = features[count][0]    
            if mz + 0.01 >= a and mz - 0.01 <= a:
                pass
            else:
                
                continue

        #if we make it back here it's a legit spectra
        filename = item[3]

        if filename == 'unavailable':
           continue
        count += 1
        fil = filename.replace('.mzXML', '_outdir').replace('.mzML', '_outdir')
        line = os.path.join('./output_nf_2', fil, 'ordered_list2.json')
       
        for spectra_info_file in data[line][dset_type]:     
            
            spectra_rt = round(spectra_info_file['retentionTime'], 6)
            spectra_mz = round(spectra_info_file['precursorMz'], 6)      
            if mz + 0.01 >= spectra_mz and mz - 0.01 <= spectra_mz:       
                if rt + 0.5 >= spectra_rt and rt - 0.5 <= spectra_rt:
                    key = spectra_info_file['key']
                    
        if key == -1:
            continue

        file_data = mzxml.MzXML(os.path.join('./test_dataset',filename)) 
         
        mz_array = file_data[key].get('m/z array').tolist()
        intensity_array = file_data[key].get('intensity array').tolist()
        seen = []
        seen_int = []
   
        for spec_inten, spec_mass in zip(spectra, mass_range):
            if spec_inten > 0:
                for actual_mass, actual_intent in zip(mz_array, intensity_array):
                    if math.floor(actual_mass) == spec_mass and actual_intent > 0:
                        
                        if math.floor(actual_mass) not in seen:
                            seen.append(math.floor(actual_mass))
                            seen_int.append(actual_intent)
                            new_inten.append(spec_inten)
                            new_mz.append(actual_mass)
                        else:
                            local = seen.index(math.floor(actual_mass))
                            if actual_intent > seen_int[local]:
                                del seen_int[local]
                                del seen[local]
                                del new_inten[local]
                                del new_mz[local]

                                seen.append(math.floor(actual_mass))
                                seen_int.append(spec_inten)
                                new_inten.append(spec_inten)
                                new_mz.append(actual_mass)
        
        new_inten =[item * intensity for item in new_inten] 
        parameters = {"FEATURE_ID": count, 'FILENAME': filename , 'PEPMASS' : str(mz), "SCANS": key, "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN" : intensity}
        spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
        spectra_list.extend(copy.deepcopy(spectra))
        return_feat.append((mz, key))
       
    if not final_param:
        mgf.write(spectra = spectra_list, output = "./%s_5.mgf" %dset_type, write_charges = False, use_numpy = True)
    else:
        mgf.write(spectra = spectra_list, output = "./predictions_5.mgf", write_charges = False, use_numpy = True)
    return(return_feat)


def main():
    filename = './test_dataset/PHT_20_018_Cort_1000.mzXML'
    #parser = argparse.ArgumentParser(description='ML process file')
    #parser.add_argument('filename')
    #args = parser.parse_args()
    
    model_name = './models/autoencoder/conv1d_22.h5'
    
    low_dset, high_dset = get_hdf5()
    print(low_dset.shape)
    print(high_dset.shape)

    predictions = predict_high(model_name, low_dset)
    mirror_plot(predictions[124], high_dset[124])
    
    #FORMAT THINGS BASED ON
    #features = format_mgf(low_dset, low_sup, "low")
    #format_mgf(predictions, low_sup, "predict", features)
    #format_mgf(high_dset, high_sup, "low", features)

    #low_peaks, high_peaks = get_npz()
    #supp_info_parse(low_peaks, high_peaks)

    #features = format_mgf_from_file(low_dset, 'low')
    #format_mgf_from_file(high_dset, 'high', features)
    #format_mgf_from_file(predictions, 'low', features, 'predictions') 
    #ormat_mgf_from_file(high_dset, high_sup)
    #ormat_mgf_from_file(predictions, low_sup)


    #data = read_mzXML(filename)
    #processed_dict, intensity_array_list = extract_spectra(data, filename) 
    #predictions = predict_high(model_name, processed_dict, intensity_array_list)
    #print(predictions.shape)

    #replace_scans(processed_dict, predictions, filename)

if __name__ == "__main__":  
    main()
