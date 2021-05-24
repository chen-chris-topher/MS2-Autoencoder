import ms2_model
import numpy as np
import h5py
import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('data', help='testing data in relative path')
parser.add_argument('model', help='select the saved model being tested and evaluated in relative path')
parser.add_argument('num_train_specs', help='number of spectra that were used in training')
args = parser.parse_args()

#the number of spectra used to train, everything else we use to tes
num_train_specs = args.num_train_specs

#full path of the data to test with
data = args.data

#name of model storage location
dirname = './models/conv1d'

#name of model to tes
model_path = os.path.join(dirname, args.model)

model = load_model(model_path)

f = h5py.File(data, 'r')
dataset_low = f['low_peaks']
dataset_high = f['high_peaks']

prediction = ms2_model.predict_model(model, dataset_low[num_train_specs:])

#evaluation = ms2_model.eval_model(model, dataset_low, dataset_high)
#print(evaluation)
#print('Testing accuracy: ', evaluation[1])

save_name = args.model.replace('.h5','')
#save predcitions as .npy
save_path = os.path.join(dirname, '%s_predictions.npy' %save_name)
np.save(save_path, prediction)
