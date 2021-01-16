import h5py
import numpy as np
from scipy.spatial.distance import cosine


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
    with h5py.File('./data_10.hdf5', 'r') as hf:
        df_low = hf['low_peaks']
        df_high = hf['high_peaks']
        
        with h5py.File('./cos_red_data_10.hdf5', 'w') as nhf:
            n_low  = nhf.create_dataset('low_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')
            n_high = nhf.create_dataset('high_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')

            for low,high in zip(df_low, df_high):
                """
                try:
                    cos = 1 - cosine(low, high)
                except:
                    continue
                if cos >= 0.5:
                """
                if np.count_nonzero(low) > 0 and np.count_nonzero(high) > 0:
                    low = np.add.reduceat(low, np.arange(0, 200000, 100))
                    high = np.add.reduceat(high, np.arange(0, 200000,100))
                    
                    size = n_low.shape
                    curr_len = size[0]
                    new_len = curr_len + 1
                    
                    n_low.resize((new_len, 2000))
                    n_high.resize((new_len, 2000))
                    
                    n_low[curr_len, :] = low
                    n_high[curr_len, :] = high
          
                    print(n_low.shape)

if __name__ == "__main__":
    concat_hdf5s()
