import ms2_model
import numpy as np
import h5py
from os.path import join
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('data', help='training data')
parser.add_argument('model', help='select the model being trained')
parser.add_argument('path', help='directory path, differs based on user/system')
parser.add_argument('--val_data', help='validation data')
parser.add_argument('data_resolution', help='bin specficity of data')
parser.add_argument('model_name', help='name to save model under')

args = parser.parse_args()
data = args.data
model = args.model
path = args.path
val_data = args.val_data
data_resolution = args.data_resolution
model_name = args.model_name

outdir = join(path, 'models/')


f = h5py.File(data, 'r')
dataset_low = f['low_peaks']
dataset_high = f['high_peaks']
       
if args.val_data:
    g = h5py.File(val_data, 'r')
    X_val = g['low_peaks']
    y_val = g['high_peaks']
else:
    print('no val data')

#ms2_model.session_config(1)
print(dataset_low.shape)
if model=='conv1d':
    model = ms2_model.model_Conv1D()
    model, loss_dict = ms2_model.fit_autoencoder(model, dataset_low, dataset_high)
    ms2_model.save_model(model, join(outdir, 'conv1d/', 'conv1d_36.h5'))
    ms2_model.save_history(loss_dict, join(outdir, 'conv1d/', 'conv1d_36_history.pickle'))

elif model=='deepautoencoder':
    autoencoder = ms2_model.model_deep_autoencoder()
    autoencoder = ms2_model.fit_model(autoencoder, dataset_high, dataset_high)
    ms2_model.save_model(autoencoder, join(outdir, 'deepautoencoder/', 'deepautoencoder.h5'))
    ms2_model.save_history(autoencoder.history, join(outdir, 'deepautoencoder/', 'deepautoencoder_history.pickle'))

elif model=='autoencoder':
    
    if data_resolution == 'low':
        autoencoder = ms2_model.initalize_autoencoder_low_res()
    if data_resolution == 'high':
        autoencoder = ms2_model.initialize_autoencoder_high_res()

    autoencoder,loss_dict = ms2_model.fit_autoencoder(autoencoder, dataset_low, dataset_high, data_resolution)    
    ms2_model.save_model(autoencoder, join(outdir, 'autoencoder/', '%s.h5' %model_name))
    ms2_model.pickle_loss_dict(loss_dict, join(outdir, 'autoencoder/', '%s_history.pickle' %model_name))

elif model=='variationalautoencoder':
    autoencoder = ms2_model.model_variational_autoencoder()
    autoencoder = ms2_model.fit_model(autoencoder, dataset_low, dataset_high)
    ms2_model.save_model(autoencoder, join(outdir, 'variationalautoencoder.h5'))

elif model=='kerastuned':
    tuner = ms2_model.keras_tune()
    tuner = ms2_model.keras_fit(tuner, dataset_low, dataset_high, X_val, y_val)
    keras_test(tuner)

print('operations complete')
