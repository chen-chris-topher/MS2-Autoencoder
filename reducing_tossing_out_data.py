import h5py
import numpy as np
from scipy.spatial.distance import cosine


def concat_hdf5s():
    file_list = ['cos_1.hdf5', 'cos_2.hdf5']
    
    with h5py.File(file_list[0], 'a') as hf:
        dset_low = hf['low_peaks']
        dset_high = hf['high_peaks']
        size = dset_low.shape
       

        with h5py.File(file_list[1], 'a') as h2:
            add_low = h2['low_peaks']
            add_high = h2['high_peaks']

            new_size = add_low.shape
            print(new_size)
            
            dset_low.resize((5044, 200000))
            dset_high.resize((5044,200000))
            
            print('resized')
            dset_low[3058:5044, :] = add_low
            dset_high[3058:5044, :] = add_high

def main():
    with h5py.File('./data_2.hdf5', 'r') as hf:
        df_low = hf['low_peaks']
        df_high = hf['high_peaks']
        
        with h5py.File('./cos_red_data_2.hdf5', 'w') as nhf:
            n_low  = nhf.create_dataset('low_peaks', shape=(1, 200000), maxshape=(None, 200000), compression='gzip')
            n_high = nhf.create_dataset('high_peaks', shape=(1, 200000), maxshape=(None, 200000), compression='gzip')

            for low,high in zip(df_low, df_high):
                print('yo')
                try:
                    cos = 1 - cosine(low, high)
                except:
                    continue
                if cos >= 0.5:
                    

                    size = n_low.shape
                    curr_len = size[0]
                    new_len = curr_len + 1
                    
                    n_low.resize((new_len, 200000))
                    n_high.resize((new_len, 200000))
                    
                    n_low[curr_len, :] = low
                    n_high[curr_len, :] = high
          
                    print(n_low.shape)

if __name__ == "__main__":
    concat_hdf5s()
