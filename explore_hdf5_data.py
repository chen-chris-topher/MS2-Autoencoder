import json
import h5py
import  matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cosine

def data_subsetting(low_dset, high_dset):
    with h5py.File('./shuffle_.hdf5', 'w') as nhf:
        n_low  = nhf.create_dataset('low_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')
        n_high = nhf.create_dataset('high_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')

        counter = 0
        for count, (low, high) in enumerate(zip(low_dset, high_dset)):
            execute = True
            cos = 1 - cosine(low, high)
            
            if cos <= 0.98:
                if np.count_nonzero(low) >= np.count_nonzero(high):
                    execute = True
                else:
                    execute = False
            else:
                execute = False
            
            """ 
            if cos > 0.97:
                counter += 1
                if counter > 20000:
                    execute = False
            """
             
            if execute is True:
                hmax = np.max(high)
                lmax = np.max(low)
                low = np.true_divide(low, lmax)
                high = np.true_divide(high, hmax)

                new_size = n_low.shape[0] + 1
                n_low.resize((new_size, 2000))
                n_high.resize((new_size, 2000))

                n_low[new_size-1, :] = low
                n_high[new_size-1, :] = high
             
            if n_low.shape[0] % 100000 == 0:
                print(n_low.shape)

            if n_low.shape[0] > 4000000:
                break
            
    #print(n_low.shape)

def cosine_distribution(low_dset, high_dset):
    import seaborn as sns

    all_cos = []
    new_cos = []
    end_cos = []
    low_array = []
    high_array = []
    count_high_more_peaks = 0
    for count,(low, high) in enumerate(zip(low_dset, high_dset)):
        if count > 7560000:
            break
        
        low_array.append(np.count_nonzero(low))
        high_array.append(np.count_nonzero(high))
        cos = 1 - cosine(low, high)
        
        minval = np.min(low[np.nonzero(low)])
        print(np.max(high))
        #if np.count_nonzero(high) > np.count_nonzero(low):
        #    count_high_more_peaks += 1
        

        """            
        if cos >= 0.70 and cos < 0.8:
            all_cos.append(cos)
        if cos >= 0.80 and cos < 0.9:
            new_cos.append(cos)
        if cos >= 0.90 and cos < 0.95:
            end_cos.append(cos)
        
        #all_cos.append(cos)
        if count % 100000 == 0:
            print(count)

        #if count > 500000:
        #    break
        """
        if str(cos) != 'nan':
            all_cos.append(cos)

    #print(len(all_cos))
    #print(len(new_cos))
    #print(len(end_cos))
    print("High ", np.average(high_array))
    print("Low ", np.average(low_array))
    #print(np.average(all_cos))
    print(np.average(all_cos))
    ax= sns.distplot(all_cos)
    plt.show()

def main():
    hf = h5py.File('./shuffle_subset_8.hdf5', 'r')
    #hf = h5py.File('./test_data/MSV000087127_nominal.hdf5', 'r')
    #hf = h5py.File('./nominal_1000.hdf5', 'r')
    #hf = h5py.File('./MSV000086270_nominal.hdf5')
    #hf = h5py.File('./test_data/test_data.hdf5', 'r')
    #hf = h5py.File('subset_data_9.hdf5', 'r')
    low_dset = hf['low_peaks']
    high_dset = hf['high_peaks']

    #print(low_dset.shape)
    #print(high_dset.shape)
    cosine_distribution(low_dset, high_dset) 
    #data_subsetting(low_dset, high_dset)
def read_scan():
    new_list = []
    with open('scan_file.txt', 'r') as hf:
        for line in hf:
            new_list.append(line.strip())
    print(len(new_list))
if __name__ == "__main__":
    read_scan()
