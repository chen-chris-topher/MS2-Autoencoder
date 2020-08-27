import tensorflow as tf

from tf.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D
from tf.keras.layers.advanced_activations import LeakyReLU
from tf.keras.models import Model
from tf.keras import backend as K
from tf.keras.callbacks import TensorBoard

import numpy as np
import pickle
import json
import h5py

def session_config(allocation=1):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=allocation)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

def generator(X_data, y_data, batch_size):
    print('generator initiated')
    steps_per_epoch = X_data.shape[0]
    number_of_batches = steps_per_epoch // batch_size
    i = 0
    
    while True:
        X_batch = X_data[i*batch_size:(i+1)*batch_size]
        y_batch = y_data[i*batch_size:(i+1)*batch_size]
        i += 1
        yield X_batch, y_batch
        print('\ngenerator yielded a batch %s' %i)
        
        if i >= number_of_batches:
            i = 0

def training_generator(X_data, y_data, batch_size):
    print('training generator initiated')
    steps_per_epoch = X_data.shape[0]
    number_of_batches = steps_per_epoch // batch_size
    i = 0
    
    while True:
        X_batch = X_data[i*batch_size:(i+1)*batch_size]
        y_batch = y_data[i*batch_size:(i+1)*batch_size]
        i += 1
        yield X_batch, y_batch
        print('\ntraining generator yielded a batch %s' %i)
        
        if i >= number_of_batches:
            i = 0

def validation_generator(X_data, y_data, batch_size):
    print('validation generator initiated')
    steps_per_epoch = X_data.shape[0]
    number_of_batches = steps_per_epoch // batch_size
    i = 0
    while True:
        X_batch = X_data[i*batch_size:(i+1)*batch_size]
        y_batch = y_data[i*batch_size:(i+1)*batch_size]
        i += 1
        yield X_batch, y_batch
        print('\nvalidation generator yielded a batch %s' %i)

        if i >= number_of_batches:
            i = 0

def test_generator(X_data, batch_size):
    print('generator initiated')
    steps_per_epoch = X_data.shape[0]
    number_of_batches = steps_per_epoch // batch_size
    i = 0

    while True: 
        X_batch = X_data[i*batch_size:(i+1)*batch_size]
        i += 1
        yield X_batch
        print('\ngenerator yielded a batch %s' %i)

        if i >= number_of_batches:
            i = 0

def fit_model(model, X_data, y_data):
    batch_size = 10
    model.fit_generator(generator=generator(X_data, y_data, batch_size),
                        max_queue_size=40, 
                        steps_per_epoch=X_data.shape[0] // batch_size, 
                        epochs=1,
                        callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
    return model

def fit_val_model(model, X_data, y_data, X_val, y_val):
    batch_size = 10000
    batch_size_val = 1000
    model.fit_generator(generator=generator(X_data, y_data, batch_size),
                        validation_data=validation_generator(X_val, y_val, batch_size_val),
                        validation_steps=X_val.shape[0],
                        steps_per_epoch=X_data.shape[0] // batch_size,
                        max_queue_size=40,
                        epochs=1)
    return model

def fit_val_model2(model, X_data, y_data):
    batch_size = 10000
    split = 0.8
    train_len = int(split * len(X_data))
    val_len = int((1 - split) * len(X_data))

    model.fit_generator(generator=generator(X_data[:train_len], y_data[:train_len], batch_size),
                        validation_data=validation_generator(X_data[train_len:], y_data[train_len:], batch_size),
                        validation_steps=val_len,
                        steps_per_epoch=train_len // batch_size,
                        max_queue_size=40,
                        epochs=1)
    return model

def predict_model(model, X_data):
    batch_size = 100
    prediction = model.predict_generator(generator=test_generator(X_data, batch_size),
                                            max_queue_size=10,
                                            steps=X_data.shape[0] // batch_size)
    return prediction

def eval_model(model, X_data, y_data):
    batch_size = 10000
    evaluation = model.evaluate_generator(generator=generator(X_data, y_data, batch_size),
                                            max_queue_size=40,
                                            steps=X_data.shape[0] // batch_size)
    return evaluation

def save_model(model, name_h5):
    model.save(name_h5)
    print('model has been saved to .h5')

def save_history(history, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    print('training history has been saved to %s' %filename)

def load_history(history_file):
    file = open(history_file)
    history_dict = pickle.load(file)
    return history_dict

def model_Conv1D():
    input_size = 2000
    input_scan = Input(shape=(input_size, 1))
    print(input_scan.shape)
    hidden_1 = Conv1D(1, (5, ), activation='relu', padding='same')(input_scan)
    print(hidden_1.shape)
    hidden_2 = MaxPooling1D()(hidden_1)
    hidden_3 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_2)
    print(hidden_3.shape)
    hidden_4 = MaxPooling1D()(hidden_3)
    hidden_5 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_4)
    print(hidden_5.shape)
    encoded = MaxPooling1D((5, ))(hidden_5)
    print(encoded.shape)

    hidden_6 = Conv1D(1, (5, ), activation='relu', padding='same')(encoded)
    print(hidden_6.shape)
    hidden_7 = UpSampling1D(5)(hidden_6)
    hidden_8 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_7)
    print(hidden_7.shape)
    hidden_9 = UpSampling1D()(hidden_8)
    hidden_10 = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_9)
    print(hidden_10.shape)
    hidden_11 = UpSampling1D()(hidden_10)
    decoded = Conv1D(1, (5, ), activation='relu', padding='same')(hidden_11)
    print(decoded.shape)

    model = Model(input_scan, decoded)
    model.compile(optimizer='adadelta', loss='cosine_proximity', metrics=['accuracy'])
    return model

def model_deep_autoencoder():
    input_size = 2000
    encoding_dim = 100
    input_scan = Input(shape=(input_size,))
    hidden_1 = Dense(1000, activation='relu')(input_scan)
    hidden_2 = Dense(500, activation='relu')(hidden_1)
    encoded = Dense(encoding_dim, activation='relu')(hidden_2)

    hidden_3 = Dense(500, activation='relu')(encoded)
    hidden_4 = Dense(1000, activation='relu')(hidden_3)
    decoded = Dense(input_size, activation='relu')(hidden_4)

    autoencoder = Model(input_scan, decoded)
    autoencoder.compile(optimizer='adadelta', loss='cosine_proximity', metrics=['accuracy'])
    return autoencoder


def model_autoencoder():
    input_size = 2000
    encoding_dim = 2000
    input_scan = Input(shape=(input_size,))
    encoded = Dense(encoding_dim, activation='relu')(input_scan)

    decoded = Dense(input_size, activation='relu')(encoded)

    autoencoder = Model(input_scan, decoded)
    autoencoder.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])
    return autoencoder

###EVERYTHING BELOW HERE IS A CHRISSY THING!
def batch_shuffle_data():
	pass

def initalize_autoencoder():
	input_size = 200000
	encoding_dim = 2000

	input_scan = Input(shape=(input_size,))
	encoded = Dense(encodind_dim, activation='relu')(input_scan)
	decoded = Dense(input_size, activation='relu')(encoded)

	autoencoder = Model(input_scan, decoded)
	autoencoder.compile(optimizer='adam', loss='cosine_similarity', metrics=['accuracy'])
	return(autoencoder)

def fit_autoencoder(autoencoder, dataset_X, dataset_Y):
	dataset = tf.Dataset.zip((dataset_x, dataset_Y))
	dataset.shuffle(
	batched_dataset = dataset.batch((200000, 32))

	for batch in batched_dataset:
		autoencoder.fit(x=batch, validation_split=0.30, epochs=50, shuffle=True, use_multiprocessing=True)
		
	return(autoencoder)







	
