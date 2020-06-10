import ms2_model
import numpy as np
import h5py
import os
import argparse

from tensorflow.keras.models import load_model

parser = argparse.ArgumentParser()
parser.add_argument('data', help='testing data in relative path')
parser.add_argument('model', help='select the saved model being tested and evaluated in relative path')

args = parser.parse_args()

dirname = os.path.dirname(os.path.realpath(__file__))
data = os.path.join(dirname, args.data)
model_path = os.path.join(dirname, args.model)

#my line
model_path = "./models_new/autoencoder/autoencoder.h5"

model = load_model(model_path)

f = h5py.File(data, 'r')
dataset_low = f['low_peaks']
dataset_high = f['high_peaks']

prediction = ms2_model.predict_model(model, dataset_high)
evaluation = ms2_model.eval_model(model, dataset_high, dataset_high)
print('Testing accuracy: ', evaluation[1])
save_path = os.path.join(dirname, 'predictions.npy')
np.save(save_path, prediction)
