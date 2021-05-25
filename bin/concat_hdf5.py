import os
import numpy as np
import h5py
big_listo = []
def extract_npz(filename):
    file = np.load(filename, allow_pickle=True)
    data = file['arr_0'] 
    return data

def remove_blank_scans(peaks_array):
    new_peaks_array = []
    for i in range(0, len(peaks_array)):
        if np.count_nonzero(i) != 0:
            new_peaks_array.append(peaks_array[i])
    return np.array(new_peaks_array)

def normalize_peaks(peaks_array, norm):
    """
    normalizes each array by dividing the array by the max of that array
    """
    from sklearn.preprocessing import normalize
    
    for i in range(0, len(peaks_array)):
        peaks_array[i] = normalize(peaks_array[i], norm=norm)
    return peaks_array

def split_reshape(data, norm):
    """
    split data into low peak and high peak sets
    outputs shape (len(data), 2000)
    """
    split_data = np.split(data, 2, axis=1) 
    low_peaks = split_data[0] 
    
    low_peaks = normalize_peaks(low_peaks, norm) #normalize
    
    low_peaks = low_peaks.reshape(len(low_peaks), np.prod(low_peaks.shape[1:]))
    high_peaks = split_data[1]
    high_peaks = normalize_peaks(high_peaks, norm) #normalize
    high_peaks = high_peaks.reshape(len(high_peaks), np.prod(high_peaks.shape[1:]))
    return low_peaks, high_peaks

def split_reshape_Conv1D(data, norm):
    """
    split data into low peak and high peak sets
    outputs shape (len(data), 2000, 1)
    """
    split_data = np.split(data, 2, axis=1)
    low_peaks = split_data[0]
    low_peaks = normalize_peaks(low_peaks, norm) #normalize
    low_peaks = low_peaks.reshape(len(low_peaks), np.prod(low_peaks.shape[1:]))
    low_peaks = np.expand_dims(low_peaks, axis=2)
    high_peaks = split_data[1]
    high_peaks = normalize_peaks(high_peaks, norm) #normalize
    high_peaks = high_peaks.reshape(len(high_peaks), np.prod(high_peaks.shape[1:]))
    high_peaks = np.expand_dims(high_peaks, axis=2)
    return low_peaks, high_peaks

#stitch all data, concatenate, save to compressed
def stitch_npz(file_list):
    data_list = []
    for filename in file_list:
        print('extracting and appending %s' %filename)
        data = extract_npz(filename)
        data_list.append(data)

    big_data = np.concatenate(data_list, axis=0) #concatenate here is going to take the most RAM 

    np.savez_compressed('concated_data', big_data)

def extra_information(file):
    import json
    filename = file.replace('ready_array2.npz','ordered_list2.json')
    with open(filename, 'r') as hf:
        data = json.load(hf)
    for item in data:
        for a in item:
            for b in a:
                filename = b['filename']
                scan = b['scan']               
                big_listo.append(filename + ' ' + str(scan))

#hdf5 stitching
def stitch_hdf5(file_list, norm, name='big_data.hdf5'):
    """
    concatenate data into hdf5 format for reading from disk
    outputs one hdf5 file with two datasets
    data in datasets will be normalized
    """

    with h5py.File(name, 'w') as f: #create empty hdf5 file with two datasets
        dataset = f.create_dataset('low_peaks', shape=(1, 2000), maxshape=(None, 2000),compression='gzip')
        dataset = f.create_dataset('high_peaks', shape=(1, 2000), maxshape=(None, 2000),compression='gzip')
        f.close()

    i_prev = 0
    len_total = 0
    count = 0
    for filename in file_list:
        extra_information(filename)
        try:
            print('#%r extracting and appending %s to hdf5' %(count, filename))
            data = extract_npz(filename)   
            size = data.shape 
            print(size)
            i_curr = size[0] + i_prev
            len_total = i_curr
            count += 1
            #print(size)
            #print(i_curr)
            #print(i_prev)
            #print(len_total)
            low_peaks, high_peaks = split_reshape(data, norm) 
            
            with h5py.File(name, 'a') as f:
                dataset = f['low_peaks'] #append to low_peaks dataset
                
                dataset.resize((len_total, 2000))
                dataset[i_prev:i_curr, :] = low_peaks

                dataset = f['high_peaks'] #append to high_peaks dataset
                dataset.resize((len_total, 2000))
                dataset[i_prev:i_curr, :] = high_peaks

                f.close()
            i_prev = i_curr
            
            print('length at %s' %len_total)
        except:
            print("Failed to reshape data into hdf5")
            pass
    with open('all_files.txt', 'a') as hf:
        for line in big_listo:
            hf.write(line + '\n')

    print('saved all data to %s' % name)

def stitch_hdf5_Conv1D(file_list, norm, name='big_data_conv1d.hdf5'):
    """
    concatenate data into hdf5 format for reading from disk
    outputs one hdf5 file with two datasets in the shape for keras.input.Conv1D
    data in datasets will be normalized
    """
    with h5py.File(name, 'w') as f: #create empty hdf5 file with two datasets
        dataset = f.create_dataset('low_peaks', 
                                    shape=(1, 20000, 1), 
                                    maxshape=(None, 20000, 1),
                                    compression='gzip')
        dataset = f.create_dataset('high_peaks', 
                                    shape=(1, 20000, 1), 
                                    maxshape=(None, 20000, 1),
                                    compression='gzip')
        f.close()

    i_prev = 0
    len_total = 0
    count = 0
    for filename in file_list:
        try:
            print('#%r extracting and appending %s to hdf5' %(count, filename))
            data = extract_npz(filename)
            size = data.shape 
            i_curr = size[0] + i_prev
            len_total += i_curr
            count += 1

            low_peaks, high_peaks = split_reshape_Conv1D(data, norm)

            with h5py.File(name, 'a') as f:
                dataset = f['low_peaks'] #append to low_peaks dataset
                dataset.resize((len_total, 200000, 1))
                dataset[i_prev:i_curr, :, :] = low_peaks

                dataset = f['high_peaks'] #append to high_peaks dataset
                dataset.resize((len_total, 200000, 1))
                dataset[i_prev:i_curr, :, :] = high_peaks

                f.close()
            i_prev = i_curr
            print('length at %s' %len_total)
        except:
            pass
    print('saved all data to %s' % name)

def get_file_list(path, data_name):
    file_list = []
    for i in os.listdir(path):
        for filename in os.listdir(os.path.join(path, i)):
            #print(filename)
            if filename  == data_name:
                file_list.append(os.path.join(path, i, filename))
                #print(os.path.join(path,filename, i, filename))
    return file_list
