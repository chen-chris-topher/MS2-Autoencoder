import h5py
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('hdf5_filename', help='input name of hdf5')
args = parser.parse_args()
filename = args.hdf5_filename
random.seed(10)

f = h5py.File('%s'%filename, 'r')
low_dset = f['low_peaks']
high_dset = f['high_peaks']

new_filename_base = filename.replace('.hdf5','')

total_rows = low_dset.shape[0]
indexes = list(range(0,total_rows))
random.shuffle(indexes)

#first we just create the file and the dataset
with h5py.File("shuffle_%s.hdf5" %new_filename_base, "w") as hf:
    dataset = hf.create_dataset('low_peaks', shape=(0, 2000), maxshape=(None, 2000),compression ='gzip')
    dataset = hf.create_dataset('high_peaks', shape=(0, 2000), maxshape=(None, 2000),compression='gzip')
    hf.close()

len_total = 0
i_prev = 0

for index in indexes:
    with h5py.File("shuffle_%s.hdf5" %new_filename_base, "a") as hf:
        dataset_low = hf['low_peaks']
        dataset_high = hf['high_peaks']
        
        #total number of rows so far
        i_curr = dataset_low.shape[0]
        
        dataset_low.resize((i_curr + 1, 2000))

        dataset_low[i_curr:i_curr + 1 , :] = low_dset[index, :]
       
        dataset_high.resize((i_curr + 1, 2000))
        dataset_high[i_curr:i_curr + 1, :] = high_dset[index, :]
        
        if dataset_low.shape[0] % 100000 == 0:
            print("Matrix Size ", dataset_low.shape)
            print("Percent Compelte ", dataset_low.shape[0]/total_row.shape[0])

