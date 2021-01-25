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


def yank_pairs():
    dir = './test_dataset/output_nf_2'
    file = 'ordered_list2.json'
    all_folders =os.listdir(dir)
    all_files = [os.path.join(dir,item, file) for item in all_folders]
    new_dict = {}
    count = 0
    for file in all_files:
        new_dict[file] = {}
        new_dict[file]['low'] = []
        new_dict[file]['high']=[]

    for file in all_files:
        if os.path.exists(file):
            with open(file) as f:
                data = json.load(f)

            for thing in data:
                if len(thing) != 0:
                    for dict in thing:
                        count += 1
                        dict[0]['filename'] = file
                        dict[1]['filename'] = file
                        new_dict[file]['low'].append(dict[0])
                        new_dict[file]['high'].append(dict[1])
    print(count)
    with open('./test_dataset/pairs.txt', 'w') as json_file:
        json.dump(new_dict, json_file)


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

def supp_info_parse(low_dset, high_dset):
    target = './ready_array.npz'
    data = np.load(target)
    data = data['arr_0']
    
    split_data = np.split(data, 2, axis=1)
    low_supp = split_data[0]
    high_supp = split_data[1]
    

    return(low_supp, high_supp)
    #format_mgf(low_dset, low_supp, 'low')
    #format_mgf(high_dset, high_supp, 'high')


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
    target = './test_dataset/hong_dataset.hdf5'
    hf = h5py.File(target, 'r')
    low_dset = hf['low_peaks']
    high_dset = hf['high_peaks']
    
    first_pass = True
    for low, high in zip(low_dset, high_dset):
        low = np.add.reduceat(low, np.arange(0, low.shape[0], 100), axis=0)        
        high = np.add.reduceat(high, np.arange(0, high.shape[0], 100), axis=0)
        
        low_max = np.max(low)
        high_max = np.max(high)
        
        if low_max == 0 or high_max == 0:
            continue

        low = np.true_divide(low, low_max)
        low[low < 0.05] = 0.0    

        high = np.true_divide(high, high_max)
        high[high < 0.05] = 0.0
        
        low_count = np.count_nonzero(low)
        high_count = np.count_nonzero(high)

        if low_count == 0 or high_count == 0:
            continue
	
        if first_pass is True:
            low_numpy = low
            high_numpy = high
            first_pass = False
        else:
            low_numpy = np.vstack((low_numpy, low))
            high_numpy = np.vstack((high_numpy, high))

    return(low_numpy,high_numpy)

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



def format_mgf_from_file(dset, sup, features=None, final_param=None):
    """
    pairs.txt 
    """
    target_file_spectra = './test_dataset/pairs.txt'
    with open(target_file_spectra) as hf:
        data = json.load(hf)

    dset_type = final_param 
    if dset_type == 'predictions':
        dset_type = 'low'
    spectra_list = [] #defines where we will save mgf appropriate spectra
    count = 0
    return_feat = []
    success_count = 0
    mass_range = range(0,2000)
    
    #loop over dataset and assoicated file names
    for spectra in dset:
        item = sup[count]
        new_inten = []
        new_mz = [] 
        rt = round(float(item[0]), 6)
        intensity = round(float(item[1]), 6)
        mz = round(float(item[2]), 6)
        key = -1 
         
        if mz == 0:
            count += 1
            continue
            
        if intensity == 0:
            count += 1
            continue

        
        """ 
        if features:
            a = float(features[count][0]) 
            if mz + 0.01 >= a and mz - 0.01 <= a:
                pass
            else:
                continue
        """

        filename = item[3]

        if filename == 'unavailable':
           continue
        count += 1
        
        fil = filename.replace('.mzXML', '_outdir').replace('.mzML', '_outdir')
        line = os.path.join('./test_dataset/output_nf_2', fil, 'ordered_list2.json')
        
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
        
        #ew_inten =[item if item > 0.05 else 0.0 for item in new_inten] 
        new_inten =[item * intensity for item in new_inten]
        
        parameters = {"FEATURE_ID": count, 'FILENAME': filename , 'PEPMASS' : str(mz), "SCANS": key, "RTINSECONDS" : rt, "CHARGE" : "1+", "MSLEVEL" : "2", "PRECURSORINTEN" : intensity}
        spectra = [{'m/z array' : new_mz, 'intensity array': new_inten, 'params': parameters}]
        spectra_list.extend(copy.deepcopy(spectra))
        return_feat.append((mz, key))
       
    if final_param == 'low' or final_param == 'high':
        mgf.write(spectra = spectra_list, output = "./%s_30.mgf" %dset_type, write_charges = False, use_numpy = True)
    else:
        mgf.write(spectra = spectra_list, output = "./predictions_30.mgf", write_charges = False, use_numpy = True)
    return(return_feat)

def sup_filename():
    all_filenames = os.listdir('./test_dataset/hong_outdir')    
    all_file_locations = [os.path.join('./test_dataset/hong_outdir', item, 'output.json') for item in all_filenames]
    
    sup_low_dict = {}
    sup_high_dict = {}
    count = 0

    for sup_file in all_file_locations:
        with open(sup_file, 'r') as jf:
            data = json.load(jf)
        filename = data['filename']
        array = data['ready_array']
       
        
        low_split = array[0:][::2]
        high_split = array[1:][::2]
        
        for low_s, high_s in zip(low_split, high_split):
            low_s.append(filename)
            high_s.append(filename)
            sup_low_dict[count] = low_s
            sup_high_dict[count] = high_s
            count += 1
           
    return(sup_low_dict, sup_high_dict) 

def main():
    
    model_name = './models/conv1d/conv1d_30.h5'
    
    low_dset, high_dset = get_hdf5()
    print(low_dset.shape)
    print(high_dset.shape)
    #redictions_0 = np.load('./gnps_predictions.npy')
    
    
    low_sup, high_sup = sup_filename()
   
    predictions = predict_high(model_name, low_dset)
    np.save('./gnps_predictions.npy', predictions)
    """
    fp =True
     
    for row in predictions_0: 
        sum_val = np.sum(row)
        row = row / sum_val
        row[row < 0.05] = 0.0
        
        if fp == True:
            fp = False
            predictions = row
        else:
            predictions = np.vstack((predictions, row))
    
    #mirror_plot(predictions[100], high_dset[100])
    #ys.exit(0)
    print(predictions)
    """
    features = format_mgf_from_file(low_dset, low_sup, None,'low')
    format_mgf_from_file(high_dset, high_sup, features, 'high')
    format_mgf_from_file(predictions, low_sup, features, 'predictions') 
    
    sys.exit(0)
    mgf_high_mz = []
    mgf_high_in = []
    with open('./high_29_3.mgf', 'r') as hf:
        content = hf.readlines()
    content = [x.strip() for x in content]
    start = False
    next_thing = False
    for item in content:
        if item == 'SCANS=3621':
            start = True    
        if next_thing is True:
            if item != 'END IONS':
                
                mgf_high_mz.append(item.split(' ')[0])
                mgf_high_in.append(item.split(' ')[1])
            else:
                next_thing = False
                start = False
        if start is True:
            if item.startswith('PRECURSORINTEN'):
                next_thing = True 
        

    mgf_predict_mz = []
    mgf_predict_in = []
    with open('./low_29_3.mgf', 'r') as hf:
        content = hf.readlines()
    content = [x.strip() for x in content]
    start = False
    next_thing = False
    for item in content:
        if item == 'SCANS=3621':
            start = True    
        if next_thing is True:
            if item != 'END IONS':
                
                mgf_predict_mz.append(item.split(' ')[0])
                mgf_predict_in.append(item.split(' ')[1])
            else:
                next_thing = False
                start = False
        if start is True:
            if item.startswith('PRECURSORINTEN'):
                next_thing = True 
    
    lib_mz = []
    lib_in = []
    with open('../indole.mgf','r') as hf:
         content = hf.readlines()
    content = [x.strip() for x in content]
    next_thing = False
    for item in content: 
        if next_thing is True:
            if item != 'END IONS':
                lib_mz.append(item.split('\t')[0])
                lib_in.append(item.split('\t')[1])
        
        if item.startswith('CHARGE'):
            next_thing = True 

    
    import spectrum_utils.plot as sup
    import spectrum_utils.spectrum as sus

    spectra = []
    print("Predict", mgf_predict_mz, mgf_predict_in)
    print("High", mgf_high_mz, mgf_high_in)
    print("Predictions Top, High Bottom")
    spectra.append(sus.MsmsSpectrum(0, 0, 0, mgf_predict_mz, mgf_predict_in,retention_time=0.0))
    spectra.append(sus.MsmsSpectrum(0, 0, 0, mgf_high_mz, mgf_high_in,retention_time=0.0))

    fig, ax = plt.subplots(figsize=(12, 6))
    spectrum_top, spectrum_bottom = spectra
    sup.mirror(spectrum_top, spectrum_bottom, ax=ax)
    plt.show()



    #data = read_mzXML(filename)
    #processed_dict, intensity_array_list = extract_spectra(data, filename) 
    #predictions = predict_high(model_name, processed_dict, intensity_array_list)
    #print(predictions.shape)

    #replace_scans(processed_dict, predictions, filename)

if __name__ == "__main__":  
    main()
