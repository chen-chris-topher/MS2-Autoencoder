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
  * *tensorflow-gpu 2.2 is what I currently use

## Structure


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

### Additional Downstream Testing
Visualizing the difference between predictions and validation data cosine scores.

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




