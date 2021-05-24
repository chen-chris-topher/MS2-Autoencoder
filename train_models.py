import ms2_model
import numpy as np
import h5py
from os.path import join
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data', help='training data')
parser.add_argument('path', help='directory path, differs based on user/system')
parser.add_argument('model_name', help='name to save model under')
parser.add_argument('num_train_specs', help='number of spectra to use for training/testing')

args = parser.parse_args()
data = args.data
path = args.path
model_name = args.model_name
num_train_spects = args.num_train_specs

outdir = join(path, 'models/')

f = h5py.File(data, 'r')
dataset_low = f['low_peaks']
dataset_high = f['high_peaks']

model = ms2_model.model_Conv1D()
model, loss_dict = ms2_model.fit_autoencoder(model, dataset_low[:num_train_specs], dataset_high[:num_train_specs], num_train_specs)
ms2_model.save_model(model, join(outdir, 'conv1d/', '%s.h5' %model_name))
ms2_model.save_history(loss_dict, join(outdir, 'conv1d/', '%s_history.pickle' %model_name))

print('operations complete')
