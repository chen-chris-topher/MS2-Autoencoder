import h5py 
import numpy as np

def main():
    with h5py.File("shuffled_data.hdf5", 'a') as hf:
        dset_low = hf['low_peaks']
        dset_high = hf['high_peaks']

        with h5py.File('./reassigned_labels.hdf5', 'w') as nhf:
            n_low  = nhf.create_dataset('low_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')
            n_high = nhf.create_dataset('high_peaks', shape=(1, 2000), maxshape=(None, 2000), compression='gzip')

            for low, high in zip(dset_low, dset_high):
                if np.count_nonzero(low) != 0 and np.count_nonzero(high) != 0:
                    size = n_low.shape
                    curr_len = size[0]
                    new_len = curr_len + 1

                    low_av = np.sum(low)/np.count_nonzero(low)
                    high_av = np.sum(high)/np.count_nonzero(high)

                    n_low.resize((new_len, 2000))
                    n_high.resize((new_len, 2000))

                    n_low[curr_len, :] = low
                    n_high[curr_len, :] = high
                    
                    if low_av > high_av:
                        n_low[curr_len, :] = high
                        n_high[curr_len, :] = low

                    else:
                        n_low[curr_len, :] = low
                        n_high[curr_len, :] = high

                    print(n_low.shape)
                



if __name__ == "__main__":
    main()
