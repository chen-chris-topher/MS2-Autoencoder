"""
Code designed to visualize 1 or more datasets with low data
high data, and predicted data
"""

import sys
#from testing_workflow_outputs import mirror_plot
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
import h5py
def read_in_actual_spectra(input_file, target_dataset):
    hf = h5py.File(input_file, 'r')
    data = hf.get(target_dataset)[:150000] 
    return(data)


def cosine_distributions(actual, low_spec, prediction=None):

    all_cos_scores = []
    real_cos_scores = []

    #used to check low versus high cosine scores prior to training
    if prediction is None:
        for high, low in zip(actual, low_spec):
            n_cos = cosine(high,low)
            real_cos_scores.append(1-n_cos)
        return(real_cos_scores)

    else:
        for pre, act, low in zip(prediction, actual, low_spec):
            n_cos = cosine(act, low)
            cos_score = cosine(act,pre)

            all_cos_scores.append(1 - cos_score)
            real_cos_scores.append(1 - n_cos)
        return(all_cos_scores)
            

parser = argparse.ArgumentParser(description='Parse hdf5 files to mgf.')
parser.add_argument('input_hdf5')
args = parser.parse_args()
input_file = args.input_hdf5


low_dset = read_in_actual_spectra(input_file, 'low_peaks')
high_dset = read_in_actual_spectra(input_file, 'high_peaks')

hf = h5py.File('./test_data/test_data.hdf5', 'r')
other_low = hf.get('low_peaks')
other_high = hf.get('high_peaks')

#predictions = np.load('predictions.npy')

#cos_score = cosine_distributions(high_dset, low_dset)
#print("Len cosine scores ", len(cos_score))
#plot_cos = [cos_score] * 3
#plot_cos = [item for items in plot_cos for item in items]

#combine_data = np.concatenate((low_dset, high_dset, predictions), axis=0)
combine_data = np.concatenate((low_dset,high_dset, other_low, other_high), axis=0)


#print(len(plot_cos))
print(low_dset.shape)
#print(predictions.shape)
print(high_dset.shape)

labels = ['low'] * low_dset.shape[0]
labels.extend(['high'] * high_dset.shape[0])
labels.extend(['other low'] * other_low.shape[0])
labels.extend(['other high'] * other_high.shape[0])
#labels.extend(['predicted'] * predictions.shape[0])

extra_labels = list(range(0,low_dset.shape[0]))
extra_labels.extend(list(range(0,high_dset.shape[0])))
#extra_labels.extend(list(range(0,predictions.shape[0])))

pca = PCA(n_components = 5)
sklearn_output = pca.fit_transform(combine_data)
print(sklearn_output.shape)


pc1 = sklearn_output[:,0]
pc2 = sklearn_output[:,1]
pc3 = sklearn_output[:,2]


g = sns.scatterplot(pc1, pc2, hue=labels)
g.set_xlabel('PC1')
g.set_ylabel('PC2')
plt.show()

#mirror_plot(low_dset[12043], high_dset[12043])

outlier_dict = {}
sys.exit(0)
for count,(x, y) in enumerate(zip(pc1, pc2)):
    if x < -0.25 and y > 0.60:
        print(x,y, count)

