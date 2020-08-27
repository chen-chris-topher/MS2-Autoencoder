# MS2-Autoencoder
MS2 Autoencoder is built on Keras for Python. The purpose of MS2 Autoencoder is to create a generalized model of MS2 spectra so that any low quality spectra can be upscaled to a high quality spectra (with quality being baed on precursor intensity). The direct general application of this tool is denoising spectra. 

## Tools
* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/distribution/)
* [NextFlow](https://www.nextflow.io/)

## Imports
* [pyteomics](https://pyteomics.readthedocs.io/en/latest/)
* [h5py](https://pypi.org/project/h5py/)
* [keras](https://keras.io/) [autoencoder tutorial](https://blog.keras.io/building-autoencoders-in-keras.html)
* [tensorflow](https://www.tensorflow.org/install/gpu) ([tensorflow-gpu](https://www.tensorflow.org/install/gpu) or [tensorflow](https://www.tensorflow.org/install)*)
  * *tensorflow-gpu worked on version 1.14 with cudnn version 10.0
  * *tensorflow-gpu 2.2 is what I currenlty use
## Structure
1. Extract mzxml/mzml files for MS2 data
2. Stitch all extracted data files (.npz) into HDF5 file (.hdf5)
3. Train autoencoder, deep autoencoder, convolutional neural network,... variational autoencoder, LSTM
4. Evaluate and predict test data on models
5. Achieve spectra upscaling/denoising

### 1. Extract mzxml
1. In MS2-Autoencoder/bin/**main.py** import extract_mzxml as em
2. The else statement in **main.py** is the entire top to bottom flow of mzxml data extraction
3. This step should be run on the cluster with nohup and NextFlow to gather all of the data
4. The Makefile includes functions (instructions) for NextFlow to run main.py on all QExactive data on GNPS(Nov/2019)
  
### 2. Stitch .npz into .hdf5
1. Use SCP to transfer extracted outdirs from cluster to local (advised that .json files are *rm -r* from outdir)
    * only **ready_array2.npz** or a .npz file is needed for stitching
2. In MS2-Autoencoder/bin/**processing.py** import concat_hdf5.py as ch5
3. Specify path to the parent directory of all outdirs, specify name of the data file ('ready_array2.npz')
4. **processing.py** will concatenate all .npz; it will output two .hdf5 files
    1. Autoencoder structured dataset
    2. Convolution neural network 1D structured dataset
    
### 3. Train models
1. Model architecture is outlined in ms2-autoencoder.py, ms2-conv1d.py, ms2-deepautoencoder.py
2. Generators, training, evaluating, predicting, and all model architectures are in ms2_model.py
3. In **train_models.py** import ms2_model.py
4. Trained models are saved as .h5 with architeture and weights
5. Models training function is built on tensorflow-gpu with gpu memory allocation and session declaration
6. Model training can be done on local or cluster machine

### 4. Evaluate and Predict models
1. Jupyter/keras load validate.ipynb is the Jupyter Notebook for loading models and visualizating predictions
2. Models prediction function is built on tensorflow-gpu with gpu memory allocation and session declaration

### 5. Spectra denoising
1. Hopefully cosine proximity is closer to 1.0 than 0.0
