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
* json


### Some Tensorflow & Cuda Notes
* Current version use tensorflow-gpu 2.3.0
    * There's a layer in the current model that requires this version or higher (this is the highest conda version)
* Cuda use is cudatoolkit 10.1
    * Should also install the correct Nvidia driver
* Cudnn install might be needed to, depends on your system specifically
* Can never get rdkit and tensorflow to play nice 
* Best of luck

### Downloading and Extracting Data
There are three ways to run this workflow.
1) Single file use
2) Small batch 1-4000 files
3) Full workflow, any number of files 

The output of extraction is 1 directory per input file containing "ready_array2.npz" and "ordered_list2.json".
An important file for small batch use and full workflow use is all_filenames.csv.
This file is a a csv saved from this link: http://dorresteintesthub.ucsd.edu:5234/datasette/database/filename.csv?_dl=on&_stream=on&_sort=filepath&sample_type__exact=GNPS&filepath__endswith=.mzML&_size=max

1) Single file use
    * Download a .mzXML or .mzML file
    * python ./bin/main_optimized.py data_file directory
    * "directory" refers to output directory
2) Small batch use
    * python wget_all_files.py lower_num upper_num
        * Downloads the data
        * lower_num and upper_num are used to slice from the file all_filenames.csv
        * An alternative is to modify this code to open any file and read filenames to download
        * This step is the bottleneck
    * nextflow run extract_data.nf -c cluster.config
        * Extracts the data
            * ready_array2.npz will be the binned data matrix with actual spectra
            * ordered_list2.json will have filename, scan number and precursor mass per spectra
            * the order is retained between these two things
        * Flag "-c cluster.config" for managing cluster use
        * Change lines in file extract_data.nf to make sure it works properly
            * params.inputSpectra = "./spectra_data_1/*{.mzXML,.mzML}"
            * params.outdir = "$baseDir/output_nf_1" 
            * Refers to location of input spectra and where extracted directories will be output 
    * python procssing.py data_path data_name
        * Concats the data into a single .hdf5 file and a .txt file
            * exports big_data.hdf5 containing two datasets "low_peaks" and "high_peaks" in order
            * exports all_files.txt with all filenames, scan numbers, and precursor masses
            * simply the stitched together extractions
        * data_path is the directory that all the extraction directories live in
            * output_nf_1 in this example 
        * data_name is the name of the .npz file with the binned data
            * defaults to ready_array2.npz in code
        * Data is L2 normalized in this step

3) Full workflow use
    * python download_extract_repeat.py
        * Only parameters that might want to be adjusted are in-line for the maxmimum number of 
        samples to be processed, default is 15,000
        * Strings togeher above workflows in chunks of 5,000 files, deleting each chunk after 
        workflow finishes
        * Numbers output files by chunk ie. nominal_1_5000.hdf5

### Training Model
1. python train_model.py 



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

### Training Data Composition Analysis
* Note that this is not a workflow, just a documentation
1. Ran library search via Proteomics2 on all files contained within the training data
2. Retrieved task ids into a .csv file './benchmark/lib_search/globals_tasks.tsv'
3. Opened every result file and compiled a json/csv with unique SpectrumID values by
running ./benchmark/lib_search/analyze_meta_workflow.py 
4. Used this csv to filter OUT ccms values for following analyses

### Spiking in Noise and Recovering Figures & Analysis
1. Generate an .hdf5 with library spectra (positive mode, binned, QE) by changing this parameter to True
    * Output is lib_spectra.hdf5 and ccms_spectra.txt
2. Run spoof_noise_recovery.py to output .npy files with noise added and predcition matrices
    * Change output file names here and here
    * Change model name here
    * Change number of spectra to predict on here
    * Tensorflow needed to run this script

Note: This is done independtly of msreduce noise workflow, but should be analagous in amount of noise
added. Meant to be able to do simultaneously.

### SIRIUS Attempt (incomplete)
1. Running Sirius in Bulk
2. Analyzing Sirius Results

### MSREDUCE Comparison
1. Generate .dta files by running benchmark/msreduce_format.py
    * This uses the file 'ccms_spectra.txt', which was generated when the library spectra were binned/downloaded
    for 'Spiking in Noise and Recovering', to find ccms ids and form them into dta files with between 1-20 peaks
    of noise spiked in
    * Make directory dta_files in benchmark folder prior to running
    * Outputs 1 dta file per ccms id per noise addition and msreduce_dta_files.txt which tracks the ccms ids
    that actually made it into dta files as sanity check
2. Move the contents of directory dta_files, msreduce_dta_files.txt, and parse_msreduce_results.py to the MSREDUCE downloaded directory
    * parse_msreduce_results.py was meant to be run from this directory
    * Make output_dir and test_file directories in MSREDUCE folder
3. Run parse_msreduce_results.py to implement msreduce and format the results properly
    * msreduce_used_files.txt is a sanity check file to make sure things got ran
    * ms_reduce.npy is a numpy matrix with binned (190000, 2000) results
    * msreduce_analysis_order.txt is the order in whch the spectra were analyzed/binned
4. Reformat results by running parse_msredcue_results.py missing_file() and make_figure() functions
    * Output useable files for autoencoder prediction and visualization
    * Documented in line in parse_msreduce_results.py and library_search.py

### Validation Data Analysis
1. Learning Curve Generation
2. Cosine Hex Plot Generation
3. Boxplot Generation
4. Viewing Model Structure
5. PCA of Data (least helpful)

### Important File Descriptions
ccms_unique_lib_search_training.csv - The list of unique ccms ids found in library search of training data
ccms_spectra.txt - The order of the library spectra

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
