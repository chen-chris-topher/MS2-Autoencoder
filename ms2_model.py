import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow as tf
import kerastuner as kt
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, InputLayer
from tensorflow.keras.layers import Activation 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l1

from scipy.spatial.distance import cosine
import numpy as np
import pickle
import json
import h5py
import random 

def session_config(allocation=1):
    print(tf.config.list_physical_devices())
    tf.test.is_built_with_cuda()
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=allocation)
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
        
        if np.count_nonzero(X_batch) == 0:
            y_batch = X_batch 
      
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
    print("Total number of batches " + str(number_of_batches))
    
    i = 0

    while True: 
        X_batch = X_data[i*batch_size:(i+1)*batch_size]
        i += 1
        print(X_batch.shape)
         

        yield X_batch
        print('\ngenerator yielded a batch %s' %i)

        if i >= number_of_batches:
            i = 0

def fit_model(model, X_data, y_data):
    batch_size = 128
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
    batch_size = 32 
    prediction = model.predict(x=test_generator(X_data, batch_size), max_queue_size=10, steps=X_data.shape[0] // batch_size)
    return prediction

def eval_model(model, X_data, y_data):
    batch_size = 128 
    evaluation = model.evaluate_generator(generator=generator(X_data, y_data, batch_size),
                                            max_queue_size=40,
                                            steps=X_data.shape[0] // batch_size)
    return evaluation

def save_model(model, name_h5):
    model.save(name_h5)
    print('model has been saved to .h5')

def save_history(history, filename):
    with open(filename, 'wb') as file_pi:
        pickle.dump(history, file_pi)
    print('training history has been saved to %s' %filename)

def load_history(history_file):
    file = open(history_file)
    history_dict = pickle.load(file)
    return history_dict

#changed aritechture to follow U-net style
def model_Conv1D():
    input_size = 2000
    input_scan = Input(shape=(input_size, 1))
   
    #layers 1-3 convolution
    hidden_1 = Conv1D(64, (3,), activation='relu', padding='same')(input_scan)
    hidden_2 = Conv1D(64, (3, ), activation='relu', padding='same')(hidden_1)
     
    max_pool_1 = MaxPooling1D(2)(hidden_2)
    print(max_pool_1.shape)

    #layers 4-6
    hidden_5 = Conv1D(128, (3, ), activation='relu', padding='same')(max_pool_1)
    hidden_6 = Conv1D(128, (3, ), activation='relu', padding='same')(hidden_5)
    
    max_pool_2 = MaxPooling1D(2)(hidden_6)

    hidden_9 = Conv1D(512, (3,), activation='relu', padding='same')(max_pool_2)
    hidden_10 = Conv1D(512, (3, ), activation='relu', padding='same')(hidden_9)
  
    up_2 = UpSampling1D(2)(hidden_10)
    concat_2 = tf.keras.layers.Concatenate(axis=2)([up_2, hidden_6])
    print("Concat 2", concat_2.shape)

    #layer 10-12
    hidden_13 = Conv1D(128, (3, ), activation='relu', padding='same')(concat_2)
    hidden_14 = Conv1D(128, (3, ), activation='relu', padding='same')(hidden_13)
    
    #upsampling layer 1 
    up_3 = UpSampling1D(2)(hidden_14)
    concat_3 = tf.keras.layers.Concatenate(axis=2)([up_3, hidden_2])

    #layer 16-18
    hidden_17 = Conv1D(64, (3, ), activation='relu', padding='same')(concat_3)
    hidden_18 = Conv1D(64, (3, ), activation='relu', padding='same')(hidden_17)
  
    decoded = Conv1D(1, (1, ), activation='relu', padding='same')(hidden_18)
    print(decoded.shape)

    sdg = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD")
    ada = tf.keras.optimizers.Adadelta(learning_rate=0.0001, rho=0.95, epsilon=1e-07, name="Adadelta")

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')


    model = Model(input_scan, decoded)
    model.compile(optimizer=adam, loss='cosine_similarity', metrics=['accuracy'])
    return model

def model_deep_autoencoder():
    input_size = 2000
    encoding_dim = 100
    input_scan = Input(shape=(input_size,))
    
    pre_hidden = Desne(2000, activation='relu')(input_scan)
    hidden_1 = Dense(1000, activation='relu')(pre_hidden)
    hidden_2 = Dense(500, activation='relu')(hidden_1)

    encoded = Dense(encoding_dim, activation='relu')(hidden_2)

    hidden_3 = Dense(500, activation='relu')(encoded)
    hidden_4 = Dense(1000, activation='relu')(hidden_3)
    post_hidden=Dense(2000, activation='relu')(hidden_4)

    decoded = Dense(input_size, activation='relu')(post_hidden)
    autoencoder = Model(input_scan, decoded)
    autoencoder.compile(optimizer='adadelta', loss='cosine_proximity', metrics=['accuracy'])
    return autoencoder


def model_autoencoder():
    input_size = 200
    encoding_dim = 2000
    input_scan = Input(shape=(input_size,))
    encoded = Dense(encoding_dim, activation='relu')(input_scan)

    decoded = Dense(input_size, activation='relu')(encoded)

    autoencoder = Model(input_scan, decoded)
    autoencoder.compile(optimizer='adam', loss='cosine_proximity', metrics=['accuracy'])
    return autoencoder

###CHRISSY CODE BELOW
###initialize an autoencoder binned nominaly
def initialize_autoencoder_low_res():
    input_size = 2000
    encoding_dim = 400

    input_scan = Input(shape=(input_size,)) 
    
    hidden_1 = Dense(1000, activation = 'relu')(input_scan)
    hidden_2 = Dense(700, activation = 'relu')(hidden_1)
    encoded = Dense(encoding_dim, activation='relu')(hidden_2)
    hidden_3 = Dense(700, activation = 'relu')(encoded)
    hidden_4 = Dense(1000, activation = 'relu')(hidden_3)
    decoded = Dense(input_size, activation='relu')(hidden_4)

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')

    autoencoder = Model(input_scan, decoded)
    autoencoder.compile(optimizer=adam, loss='cosine_similarity', metrics=['accuracy'])
   
    return(autoencoder)


def fit_autoencoder(autoencoder, X_data, y_data):   
    batch_size = 256
    split = 0.5
    test_size = int(batch_size * (1-split))
    epochs = 25
    idx = X_data.shape[0]

    test_loss = []
    train_loss = []
    i = 0
    list_of_indices = []
    while i < idx:
        temp_tuple = (i, i+batch_size+test_size)
        list_of_indices.append(temp_tuple)
        i += batch_size+test_size

    training_indices = []
    testing_indices = []
    for index_set in list_of_indices:
        training_indices.append((index_set[0], index_set[0] + batch_size))
        testing_indices.append((index_set[0] + batch_size, index_set[1]))

    for epoch in range(epochs): 
        print("\nStart of epoch %d" % (epoch,))
        list_of_indices = []
        
        random.seed(10)
        random.shuffle(training_indices)
        random.shuffle(testing_indices)
        
        val_loss = []
        acc_loss = []
        for step, train_indices in enumerate(training_indices):
            start_train = train_indices[0]
            end_train = train_indices[1]
          
            train_dict = autoencoder.train_on_batch(X_data[start_train:end_train], y_data[start_train:end_train], return_dict=True)
            val_loss.append(train_dict['loss'])
      
        for test_indices in training_indices:
            start_test = test_indices[0]
            end_test = test_indices[1]
            test_dict = autoencoder.test_on_batch(X_data[start_test:end_test], y_data[start_test:end_test], return_dict=True)
            acc_loss.append(test_dict['loss'])

        test_loss.append(np.mean(acc_loss))
        train_loss.append(np.mean(val_loss))

        print('Validation Loss: ' + str(np.mean(val_loss)))
        print('Training Loss: ' + str(np.mean(acc_loss)))
        if epoch == 0:
            print(autoencoder.summary())

    loss_dict = {'test_loss':test_loss, 'train_loss':train_loss}
    return(autoencoder, loss_dict)

def pickle_loss_dict(loss_dict, model_name):
    with open(model_name, 'wb') as handle:
        pickle.dump(loss_dict, handle)
