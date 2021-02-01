import h5py
import numpy as np
from scipy.spatial.distance import cosine


def test_data():
    filename = 'shuffled_data_zeros.hdf5'
    
    #open the original file
    with h5py.File(filename, 'r') as hf:
        dset_low = hf['low_peaks']
        dset_high = hf['high_peaks']
        
        for low in dset_low:
            for item in low:
                if item != 0:
                    print(item)
    
def get_rid_blank_spectra():
    filename = 'shuffled_data_zeros_1.hdf5'
    
    #open the original file
    with h5py.File(filename, 'a') as hf:
        dset_low = hf['low_peaks']
        dset_high = hf['high_peaks']
        
        with h5py.File('./shuffled_data_zeros_2.hdf5', 'w') as nhf:
            n_low  = nhf.create_dataset('low_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')
            n_high = nhf.create_dataset('high_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')

            for low, high in zip(dset_low, dset_high):
                low_max = max(low)
                high_max = max(high)

                low = np.true_divide(low, low_max).tolist()
                high = np.true_divide(high, high_max).tolist()
                
                low =[item if item >0.05 else 0.0 for item in low]
                high = [item if item >0.05 else 0.0 for item in high]
                
                
                cos = 1 - cosine(low, high)            
                if cos < 0.7:
                    continue

                if np.count_nonzero(low) != 0 and np.count_nonzero(high) != 0:
                    size = n_low.shape
                    curr_len = size[0]
                    new_len = curr_len + 1
                        
                    n_low.resize((new_len, 2000))
                    n_high.resize((new_len, 2000))
                        
                    n_low[curr_len, :] = low
                    n_high[curr_len, :] = high
                    
                    print(n_low.shape)


def concat_hdf5s():
    file_list = ['shuffled_data.hdf5', 'cos_red_data_10.hdf5']
    
    #open the original file
    with h5py.File(file_list[0], 'a') as hf:
        dset_low = hf['low_peaks']
        dset_high = hf['high_peaks']
        
        #open the new file
        with h5py.File(file_list[1], 'a') as h2:
            add_low = h2['low_peaks']
            add_high = h2['high_peaks']
            
            for low, high in zip(add_low, add_high):
                
                new_size = dset_low.shape[0] + 1
                
                dset_low.resize((new_size, 2000))
                dset_high.resize((new_size,2000))
                
                
                dset_low[new_size-1, :] = low
                dset_high[new_size-1, :] = high

                print(dset_low.shape)

def main():
    with h5py.File('./test_dataset/hong_dataset.hdf5', 'a') as hf:
        df_low = hf['low_peaks']
        df_high = hf['high_peaks']
        
        with h5py.File('./hong_reduced.hdf5', 'a') as nhf:
            n_low  = nhf.create_dataset('low_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')
            n_high = nhf.create_dataset('high_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')

            for low,high in zip(df_low, df_high):
                low = np.add.reduceat(low, np.arange(0, 200000, 100))
                high = np.add.reduceat(high, np.arange(0, 200000,100))

                l_max = np.max(low, axis=0)
                h_max = np.max(high, axis =0)

                if l_max > 0 and h_max > 0:
                    low = np.true_divide(low, l_max)
                    high = np.true_divide(high, h_max)

                    high[high < 0.05] = 0
                    low[low < 0.05] = 0

                    size = n_low.shape
                    curr_len = size[0]
                    new_len = curr_len + 1
                    
                    n_low.resize((new_len, 2000))
                    n_high.resize((new_len, 2000))
                    
                    n_low[curr_len, :] = low
                    n_high[curr_len, :] = high
          
                    print(n_low.shape)

if __name__ == "__main__":
    main()
