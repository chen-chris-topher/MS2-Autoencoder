import h5py
import random

f = h5py.File('new_data.hdf5', 'r')
low_dset = f['low_peaks']
high_dset = f['high_peaks']

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = numpy.empty(a.shape, dtype=a.dtype)
    shuffled_b = numpy.empty(b.shape, dtype=b.dtype)
    permutation = numpy.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


total_rows = low_dset.shape[0]
indexes = list(range(0,total_rows))
random.shuffle(indexes)

#first we just create the file and the dataset
with h5py.File("shuffled_data.hdf5", "w") as hf:
    dataset = hf.create_dataset('low_peaks', shape=(0, 2000), maxshape=(None, 2000),compression ='gzip')
    dataset = hf.create_dataset('high_peaks', shape=(0, 2000), maxshape=(None, 2000),compression='gzip')
    hf.close()

len_total = 0
i_prev = 0
for index in indexes:
    with h5py.File("shuffled_data.hdf5", "a") as hf:
        dataset_low = hf['low_peaks']
        dataset_high = hf['high_peaks']
        
        #total number of rows so far
        i_curr = dataset_low.shape[0]
        
        dataset_low.resize((i_curr + 1, 2000))

        dataset_low[i_curr:i_curr + 1 , :] = low_dset[index, :]
       
        dataset_high.resize((i_curr + 1, 2000))
        dataset_high[i_curr:i_curr + 1, :] = high_dset[index, :]

        print(dataset_low.shape)

