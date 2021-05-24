import h5py
import numpy as np
from scipy.spatial.distance import cosine


def concat_hdf5s():
    file_list = ['./nominal_1000.hdf5', './nominal_83000_84000.hdf5']
    
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
                
                if dset_low.shape[0] % 10000 == 0:
                    print(dset_low.shape)

if __name__ == "__main__":
    concat_hdf5s()
