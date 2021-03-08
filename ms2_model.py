import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, InputLayer, Conv1DTranspose, Dropout, Flatten
from tensorflow.keras.layers import Activation 
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.regularizers import l1
from tensorflow.keras.metrics import CosineSimilarity
from scipy.spatial.distance import cosine
import numpy as np
import time
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
      

        X_batch = np.add.reduceat(X_batch, np.arange(0, X_batch.shape[1], 100), axis=1)
        y_batch = np.add.reduceat(y_batch, np.arange(0, y_batch.shape[1], 100),axis=1) 
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
        
        #X_batch = np.add.reduceat(X_batch, np.arange(0, 200000, 100), axis =1)

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
    batch_size = 256 
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
        pickle.dump(history.history, file_pi)
    print('training history has been saved to %s' %filename)

def load_history(history_file):
    file = open(history_file)
    history_dict = pickle.load(file)
    return history_dict



def model_Conv1D_lowres():
    input_scan = Input(shape=(2000,1))
    
    #2,000
    hidden_1 = Conv1D(32, (3,), activation='relu', padding='same', activity_regularizer='l1')(input_scan)
    hidden_2 = Conv1D(32, (3,), activation='relu', padding='same', activity_regularizer='l1')(hidden_1)
    max_pool_1 = MaxPooling1D(2)(hidden_2)
    print("Max Pool 1", max_pool_1.shape)

    #1,000
    hidden_3 = Conv1D(64, (3,), activation='relu', padding='same', activity_regularizer='l1')(max_pool_1)
    hidden_4 = Conv1D(64, (3,), activation='relu', padding='same', activity_regularizer='l1')(hidden_3)
    max_pool_2 = MaxPooling1D(2)(hidden_4)
    print("Max Pool 2", max_pool_2.shape)

    #500 BOTTLENECK
    hidden_5 = Conv1D(128, (3, ), activation='relu', padding='same', activity_regularizer='l1')(max_pool_2) 
    hidden_6 = Conv1D(128, (3, ), activation='relu', padding='same', activity_regularizer='l1')(hidden_5)
    print("Max Pool 6", hidden_6.shape)

    #1,000
    up_2 = UpSampling1D(2)(hidden_6)
    conv_up_2 = Conv1DTranspose(64, (2, ), activation='relu', padding='same')(up_2)
    concat_2 = tf.keras.layers.Concatenate(axis=2)([conv_up_2, hidden_4])
    print("Concat 2", concat_2.shape)
 
    hidden_11 = Conv1D(64, (3, ), activation='relu', padding='same', activity_regularizer='l1')(concat_2)
    hidden_12 = Conv1D(64, (3, ), activation='relu', padding='same', activity_regularizer='l1')(hidden_11)
    print("Hidden 12", hidden_12.shape)

    #2,000
    up_3 = UpSampling1D(2)(hidden_12)
    conv_up_3 = Conv1DTranspose(64, (2, ), activation='relu', padding='same')(up_3)
    concat_3 = tf.keras.layers.Concatenate(axis=2)([conv_up_3, hidden_2])
    print("Concat 3", concat_2.shape)

    hidden_13= Conv1D(32, (3, ), activation='relu', padding='same', activity_regularizer='l1')(concat_3)
    hidden_14= Conv1D(32, (3, ), activation='relu', padding='same', activity_regularizer='l1')(hidden_13)
    print("Hidden 14", hidden_14.shape)

    decoded= Conv1D(1, (2, ), activation='relu', padding='same', activity_regularizer='l1')(hidden_14) #change inpuyt back to normal shape
    print("Decoded", decoded.shape)

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    metric = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=1)
    model = Model(input_scan, decoded)
    
    model.compile(optimizer=adam, loss=cosine_loss, metrics=[metric])
    return model


def model_Conv1D():
    input_scan = Input(shape=(200000,1))
    
    hidden_1 = Conv1D(32, (10,), strides = 10, activation='relu', padding='same')(input_scan)
    hidden_2 = Conv1D(64, (10,), strides = 10, activation='relu', padding='same')(hidden_1)
    print("Hidden 2", hidden_2.shape)

    #2,000
    hidden_3 = Conv1D(128, (3,), activation='relu', padding='same')(hidden_2)
    hidden_4 = Conv1D(128, (3,), activation='relu', padding='same')(hidden_3)
    max_pool_1 = MaxPooling1D(2)(hidden_4)
    print("Max Pool 1", max_pool_1.shape)

    #1,000
    hidden_5 = Conv1D(256, (3,), activation='relu', padding='same')(max_pool_1)
    hidden_6 = Conv1D(256, (3,), activation='relu', padding='same')(hidden_5)
    max_pool_3 = MaxPooling1D(2)(hidden_6)
    print("Max Pool 3", max_pool_3.shape)

    #500
    hidden_7 = Conv1D(512, (3, ), activation='relu', padding='same')(max_pool_3) 
    hidden_8 = Conv1D(512, (3, ), activation='relu', padding='same')(hidden_7)
    print("Hidden 8", hidden_8.shape)

    #1,000  
    up_1 = UpSampling1D(2)(hidden_8)
    conv_up_1 = Conv1DTranspose(256, (2, ), activation='relu', padding='same')(up_1)
    concat_1 = tf.keras.layers.Concatenate(axis=2)([conv_up_1, hidden_6])
    print("Concat 1", concat_1.shape)

    #2,000
    hidden_9 = Conv1D(256, (3, ), activation='relu', padding='same')(concat_1)
    hidden_10 = Conv1D(256, (3, ), activation='relu', padding='same')(hidden_9)
    print("Hidden 10", hidden_10.shape)

    #2,000
    up_2 = UpSampling1D(2)(hidden_10)
    conv_up_2 = Conv1DTranspose(128, (2, ), activation='relu', padding='same')(up_2)
    concat_2 = tf.keras.layers.Concatenate(axis=2)([conv_up_2, hidden_4])
    print("Concat 2", concat_2.shape)

    hidden_11= Conv1D(128, (3, ), activation='relu', padding='same')(concat_2)
    hidden_12= Conv1D(128, (3, ), activation='relu', padding='same')(hidden_11)
    print("Hidden 12", hidden_12.shape)

    #20,000
    up_5 = UpSampling1D(10)(hidden_12)
    hidden_13 = Conv1D(64, (3, ), activation='relu', padding='same')(up_5)
    print("Hidden 13", hidden_13.shape)

    #200,000
    up_6 = UpSampling1D(10)(hidden_13)
    hidden_14 = Conv1D(32, (3, ), activation='relu', padding='same')(up_6)
    print("Hidden 23", hidden_14.shape)

    decoded= Conv1D(1, (2, ), activation='relu', padding='same')(hidden_14) #change inpuyt back to normal shape
    print("Decoded", decoded.shape)

    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    metric = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=1)
    model = Model(input_scan, decoded)
    
    model.compile(optimizer=adam, loss=cosine_loss, metrics=[metric])
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

def fit_autoencoder(autoencoder, X_data, y_data):    
    batch_size = 512 
    split = 0.5
    test_size = int(batch_size * (1-split))
    epochs = 5
    idx = X_data.shape[0]

    print("Total X", X_data.shape)
    print("Total y", y_data.shape)

    test_loss = []
    train_loss = []
    test_acc = []
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
        t0 = time.time()
        print("\nStart of epoch %d" % (epoch,))
        list_of_indices = []
        
        random.seed(10)
        random.shuffle(training_indices)
        random.shuffle(testing_indices)
        
        val_loss = []
        acc_loss = []
        cos = []
        for step, train_indices in enumerate(training_indices):
            start_train = train_indices[0]
            end_train = train_indices[1]
            
            if step == 0 and epoch == 0:
                print("Data shape", X_data[start_train:end_train].shape)
                print("Data shape y", y_data[start_train:end_train].shape)
            train_dict = autoencoder.train_on_batch(X_data[start_train:end_train], y_data[start_train:end_train], return_dict=True)
            val_loss.append(train_dict['loss'])
      
        for test_indices in training_indices:
            start_test = test_indices[0]
            end_test = test_indices[1]
            test_dict = autoencoder.test_on_batch(X_data[start_test:end_test], y_data[start_test:end_test], return_dict=True)
            acc_loss.append(test_dict['loss'])
            cos.append(test_dict['cosine_similarity'])
            
        test_acc.append(np.mean(cos))
        test_loss.append(np.mean(acc_loss))
        train_loss.append(np.mean(val_loss))
        print('Cosine Similarity Test ' + str(np.mean(cos)))
        print('Validation Loss: ' + str(np.mean(val_loss)))
        print('Training Loss: ' + str(np.mean(acc_loss)))
        
        if epoch == 0:
            print(autoencoder.summary())
        
        t1 = time.time()

        print("Total Epoch Time: ", t1-t0)
    loss_dict = {'test_loss':test_loss, 'train_loss':train_loss, 'test_acc' : test_acc}
    return(autoencoder, loss_dict)

def pickle_loss_dict(loss_dict, model_name):
    with open(model_name, 'wb') as handle:
        pickle.dump(loss_dict, handle)
