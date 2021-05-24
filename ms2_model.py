import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, InputLayer, Conv1DTranspose, Dropout, Flatten, BatchNormalization
from tensorflow.keras.layers import Activation, LeakyReLU 
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
    """
    Paramters:
        allocation (int) : amount of memory allowed per process on gpu

    Original code meant to expicitly dedicate memory prior
    to training. Not in use - GPU does it automatically.
    """
    print("Possible training hardware", tf.config.list_physical_devices())
    
    #this is handy for checking if CUDA is properly installed
    tf.test.is_built_with_cuda()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=allocation)
    config = tf.ConfigProto(gpu_options=gpu_options)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

def generator(X_data, y_data, batch_size):
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
    batch_size = 1 
    prediction = model.predict(x=test_generator(X_data, batch_size), max_queue_size=10, steps=X_data.shape[0] // batch_size)
    return prediction

def eval_model(model, X_data, y_data):
    batch_size = 256 
    evaluation = model.evaluate(generator(X_data, y_data, batch_size),
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

def model_Conv1D():
    input_scan = Input(shape=(2000,1))    
    init = tf.keras.initializers.Orthogonal(seed=10)
    act =tf.keras.regularizers.l1(0.000001) 

    #2,000
    hidden_1 = Conv1D(32, (5,), strides = 1, padding='same', kernel_initializer = init)(input_scan)
    act_1 = tf.keras.activations.tanh(hidden_1)    
    hidden_2 = Conv1D(32, (5,), strides = 1, padding='same', kernel_initializer = init)(act_1)
    act_2 = tf.keras.activations.tanh(hidden_2)
    max_pool_1 = MaxPooling1D(2)(act_2)
    print(max_pool_1.shape)
    
    #1,000
    hidden_5 = Conv1D(64, (5,), strides = 1, padding='same', kernel_initializer = init)(max_pool_1)
    act_5 = tf.keras.activations.tanh(hidden_5)
    hidden_6 = Conv1D(64, (5,), strides = 1, padding='same', kernel_initializer = init)(act_5) 
    act_6 = tf.keras.activations.tanh(hidden_6)
    max_pool_2 = MaxPooling1D(2)(act_6)
    print(max_pool_2.shape)

    #500
    hidden_9 = Conv1D(128, (5, ), strides = 1, padding='same', kernel_initializer = init)(max_pool_2)
    act_9 = tf.keras.activations.tanh(hidden_9) 
    hidden_10= Conv1D(128, (5, ), strides = 1, padding='same', kernel_initializer = init)(act_9) 
    act_10 = tf.keras.activations.tanh(hidden_10)
    max_pool_3 = MaxPooling1D(2)(act_10)
    print(max_pool_3.shape)

    #250
    hidden_13 = Conv1D(256, (5, ), strides = 1, padding='same', kernel_initializer = init)(max_pool_3)
    act_13 = tf.keras.activations.tanh(hidden_13)
    hidden_14 = Conv1D(256, (5, ), strides = 1, padding='same', kernel_initializer = init)(act_13) 
    act_14 = tf.keras.activations.tanh(hidden_14)
    print("Act 14 ", act_14.shape)
    #500
    conv_up_1 = Conv1DTranspose(128, (5, ), strides = 2, padding='same', kernel_initializer = init)(act_14)
    act_up_1 = tf.keras.activations.tanh(conv_up_1)
    concat_1 = tf.keras.layers.Concatenate(axis=2)([act_up_1, act_10])
    hidden_25 = Conv1D(128, (5, ), strides = 1,padding='same', kernel_initializer = init)(concat_1)
    act_25 = tf.keras.activations.tanh(hidden_25)
    hidden_26 = Conv1D(128, (5, ), strides = 1, padding='same', kernel_initializer = init)(act_25) 
    act_26 = tf.keras.activations.tanh(hidden_26)
    print("Act 26 ", act_26.shape) 
    #1,000
    conv_up_2 = Conv1DTranspose(64, (5, ), strides = 2, padding='same', kernel_initializer = init)(act_26)
    act_up_2 = tf.keras.activations.tanh(conv_up_2)
    concat_2 = tf.keras.layers.Concatenate(axis=2)([act_up_2, act_6])
    hidden_29 = Conv1D(64, (5, ), strides = 1, padding='same', kernel_initializer = init)(concat_2)
    act_29 = tf.keras.activations.tanh(hidden_29)
    hidden_30 = Conv1D(64, (5, ), strides = 1, padding='same', kernel_initializer = init)(act_29)
    act_30 = tf.keras.activations.tanh(hidden_30)
    print("Act 30 ", act_30.shape)

    #2,000
    conv_up_3 = Conv1DTranspose(32, (5, ), strides = 2, padding='same', kernel_initializer = init)(act_30)
    act_up_3 = tf.keras.activations.tanh(conv_up_3)
    concat_3 = tf.keras.layers.Concatenate(axis=2)([act_up_3, act_2]) 
    hidden_33 = Conv1D(32, (5, ), strides = 1, padding='same', kernel_initializer = init)(concat_3)
    act_33 = tf.keras.activations.tanh(hidden_33)
    hidden_34 = Conv1D(32, (5, ), strides = 1, padding='same', kernel_initializer = init)(act_33)  
    act_34 = tf.keras.activations.tanh(hidden_34)
    print("Act 34 ", act_34.shape)
    decoded = Conv1D(1, (5, ), activation='tanh', padding='same', kernel_initializer = init)(act_34)
    
    print("Decoded ", decoded.shape)

    adam = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
    name='Adam')
    
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=1)
    metric = tf.keras.metrics.CosineSimilarity(name='cosine_similarity', dtype=None, axis=1)
    model = Model(input_scan, decoded)
    
    model.compile(optimizer=adam, loss=cosine_loss, metrics=[metric])
    return model

  
def fit_autoencoder(autoencoder, X_data, y_data):    
    batch_size = 512
    test_size = 32 
    print("Test Size ", test_size)
    print("Batch Size ", batch_size)
    epochs = 6 
    print("Epcohs ", epochs)
    idx = X_data.shape[0]

    print("Total X", X_data.shape)
    print("Total y", y_data.shape)

    test_loss = []
    train_loss = []
    test_acc = []
    train_acc = []
    i = 0
    list_of_indices = []

    while i+batch_size+test_size < 3300000 and i+batch_size+test_size < idx:
        temp_tuple = (i, i+batch_size+test_size)
        list_of_indices.append(temp_tuple)
        i += batch_size+test_size

    for c, thing in enumerate(list_of_indices):
        if c < 10:
            print(thing)

    training_indices = []
    testing_indices = []

    for index_set in list_of_indices:
        training_indices.append((index_set[0], index_set[0] + batch_size))
        testing_indices.append((index_set[0] + batch_size, index_set[1]))
        
    for epoch in range(epochs): 
        t0 = time.time()
        print("\nStart of epoch %d" % (epoch + 1,))
        list_of_indices = []
        
        random.seed(10)
        random.shuffle(training_indices)
        random.shuffle(testing_indices)
        
        val_loss = []
        acc_loss = []
        cos = []
        t_cos = []
        for step, train_indices in enumerate(training_indices):
            start_train = train_indices[0]
            end_train = train_indices[1]
            
            if step == 0 and epoch == 0:
                print(start_train, end_train)
                print("Data Shape Train", X_data[start_train:end_train].shape)
                
            
            train_x = np.expand_dims(X_data[start_train:end_train], axis=2)
            train_y = np.expand_dims(y_data[start_train:end_train], axis=2)
            train_dict = autoencoder.train_on_batch(train_x, train_y, return_dict=True)
            val_loss.append(train_dict['loss'])
            t_cos.append(train_dict['cosine_similarity'])

        for step, test_indices in enumerate(testing_indices):
            start_test = test_indices[0]
            end_test = test_indices[1]

            if step == 0 and epoch == 0:
                print("Data shape test", X_data[start_test:end_test].shape)
            test_x = np.expand_dims(X_data[start_test:end_test], axis =2)
            test_y = np.expand_dims(y_data[start_test:end_test], axis = 2)

            test_dict = autoencoder.test_on_batch(test_x, test_y, return_dict=True)
            acc_loss.append(test_dict['loss'])
            cos.append(test_dict['cosine_similarity'])
            
        test_acc.append(np.mean(cos))
        train_acc.append(np.mean(t_cos))
        test_loss.append(np.mean(acc_loss))
        train_loss.append(np.mean(val_loss))
        print('Train Cosine Similarity ' + str(np.mean(t_cos)))
        print('Test Cosine Similarity ' + str(np.mean(cos)))
        print('Train Loss: ' + str(np.mean(val_loss)))
        print('Test Loss: ' + str(np.mean(acc_loss)))
        
        if epoch == 0:
            print(autoencoder.summary())
        
        t1 = time.time()

        print("Total Epoch Time: ", t1-t0)
    loss_dict = {'test_loss':test_loss, 'train_loss':train_loss, 'test_acc' : test_acc, "train_acc": train_acc}
    return(autoencoder, loss_dict)

def pickle_loss_dict(loss_dict, model_name):
    with open(model_name, 'wb') as handle:
        pickle.dump(loss_dict, handle)
