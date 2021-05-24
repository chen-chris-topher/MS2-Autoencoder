import ms2_model
import numpy as np
import h5py
from os.path import join
import argparse
from scipy.spatial.distance import cosine

parser = argparse.ArgumentParser()
parser.add_argument('data', help='training data')
parser.add_argument('path', help='directory path, differs based on user/system')
parser.add_argument('model_name', help='name to save model under')

args = parser.parse_args()
data = args.data
path = args.path
classifier = args.classifier
model_name = args.model_name

outdir = join(path, 'models/')

f = h5py.File(data, 'r')
dataset_low = f['low_peaks']
dataset_high = f['high_peaks']

model = ms2_model.model_Conv1D()
model, loss_dict = ms2_model.fit_autoencoder(model, dataset_low, dataset_high)
ms2_model.save_model(model, join(outdir, 'conv1d/', 'conv1d_42.h5'))
ms2_model.save_history(loss_dict, join(outdir, 'conv1d/', 'conv1d_42_history.pickle'))

print('operations complete')
