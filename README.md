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
* scipy
* numpy
* time

### 1. Gather and Extract Data
1. Generate list of elible spectra
    1. Download the file from this link (current use positive QE data)
2. Get spectra by running a python script here
    1. Change the parameters for the number of spectra you want to download here
        * It's not realistic to download all spectra at once for memory reasons, I tend to do 4k
        at a time
        * This is the extraction bottleneck
3. In MS2-Autoencoder/bin/**main_optimized.py** import extract_mzxml as em
    1. Change the config file to fit needs here
    2. Make sure the location of the input/output is correct here
    3. If you would like to record the filename/scan number being used you need to modify this line
        * There are a bunch of paramters you can set to output different files, most of them aren't helpful
        unless you think the code is messed up
4. Run nextflow and watch data extract
    1. nextflow run extract_mzxml.nf -c cluster.config
        * The work folder gets large, in between runs it's helpful to delete it        

### 2. Stitch .npz into .hdf5
1. Run **processing.py**, which  will concatenate all .npz; it will output two files
    1. Specify path to the parent directory of all outdirs, specify name of the data file ('ready_array2.npz')
    2. This will output an .hdf5 file containing all data and .txt file containing the filepath and scan number for data tracking
    3. Note that this will also L2 normalize all data (THIS IS THE WAY WE FEED THE DATA TO THE MODEL)
        * It's a very easy transition between base-peak and l2 normalization
        * I used base-peak normalization for data analysis (mirror plots will scale this way anyways, easier to diagnose)
** If you'd like to do all of the above at the same time, check all the parameters listed, then run 
script download_extract_repeat.py (https://github.com/laserc/MS2-Autoencoder/blob/chrissys_branch_3/download_extract_repeat.py) 

### 3. Shuffle Data
1. Run shuffle_hdf5.py with the name of the .hdf5 file as a command line parameter
    1. This will create a new file, with the word "shuffle" before the filename

### 4. Train models
1. Model architecture is outlined in ms2_model.py starting at this line
    1. Current implementation is a U-Net
    2. Change weight initialization here, loss function here, and optimizer/learning rate here
2. Define the number of epochs, batch size and test size here, here, and here respectively
    1. Manually set these each time because it's fairly easy and I like to double check it
    2. TRAINING, VALIDATION, AND TESTING DATA ALL COME FROM THE SAME HDF5 FILE.
        * This means when you run the training, you will specify the lump sum of training / testing in command line
            * IE. Run train_models.py passing 3,000,000 to take that amount from the hdf5 into training
            * Splitting between training / testing happens after and ratio based on sizes set in-line
3. Code for actually running training is train_models.py
    * Specify model name to save under and number of spectra to use
    * Trained models are saved as .h5 with architeture and weights
    * Loss and accuracy history are saved in .pickle format
4. Predictions are done by running test_models.py
    * Specify model name to test 
    * Specify th number of spectra that were used in the training process,
    everything else is used to test
    * Predictions are saved to numpy matrix with _predictions.npy at the end of the model name

### Some Tensorflow & Cuda Notes
* Current version use tensorflow-gpu 2.3.0
    * There's a layer in the current model that requires this version or higher (this is the highest conda version)
* Cuda use is cudatoolkit 10.1
    * Should also install the correct Nvidia driver
* Cudnn install might be needed to, depends on your system specifically
* Can never get rdkit and tensorflow to play nice 
* Best of luck

### Downstream Testing
Visualizing the difference between predictions and validation data cosine scores.
This represents the bulk of the code, and while this will be a brief overview, a lot
of this is well-documented in-line. No command line parameters are passed in these
scripts, just things changed in-line.

### First Pass Prediction Analysis
1. Run testing_workflow_outputs.py to test/visualize predictions
    * Specify model target, hdf5 target, prediction target, and loss history in line here
    * Can use this to visualize the loss/accuracy chart, cosine hex plot and boxplots
    * Can also visualize the model summary in this function
    * Can visualize mirror plots by specifying spectra numbers here

### Spiking in Noise and Recovering Figures & Analysis
1. Cosine Distribution
2. Boxplots
3. Percent of Noise Recovered / Removed
4. Barplot on Relative Success

### SIRIUS Attempt (incomplete)
1. Running Sirius in Bulk
2. Analyzing Sirius Results

### MSREDUCE Comparison
1. Cosine Distribution
2. Boxplots

### Validation Data Analysis
1. Learning Curve Generation
2. Cosine Hex Plot Generation
3. Boxplot Generation
4. Viewing Model Structure
5. PCA of Data (least helpful)

### File Location on GNPS
1. .hdf5 data file
2. .mgf's for predicted, noisy, and library spectra
    1. Supporting .tsv and .csv files with annotations, peak counts, and noise counts
3. Model and model history
4. .tsv file for libary search for all training/testing/validation data
5. List of all filenames that fit search criteria (positive mode, QE) in use order

### Task IDs for GNPS Jobs
predicted - f02720d3e2024f5caa3deba25e73556f
noisy - 900a609a912e40728d04156a28491841
library spectra - 90dd734d44e244deb2594cf6f9734784
