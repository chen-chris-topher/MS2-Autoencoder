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
    data = hf.get(target_dataset).value 
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
predictions = np.load('predictions.npy')

cos_score = cosine_distributions(high_dset, low_dset, predictions)
print("Len cosine scores ", len(cos_score))
plot_cos = [cos_score] * 3
plot_cos = [item for items in plot_cos for item in items]

combine_data = np.concatenate((low_dset, high_dset, predictions), axis=0)
#combine_data = np.concatenate((low_dset,high_dset), axis=0)

print(len(plot_cos))
print(low_dset.shape)
print(predictions.shape)
print(high_dset.shape)

labels = ['low'] * low_dset.shape[0]
labels.extend(['high'] * high_dset.shape[0])
labels.extend(['predicted'] * predictions.shape[0])

extra_labels = list(range(0,low_dset.shape[0]))
extra_labels.extend(list(range(0,high_dset.shape[0])))
extra_labels.extend(list(range(0,predictions.shape[0])))

pca = PCA(n_components = 5)
sklearn_output = pca.fit_transform(combine_data)
print(sklearn_output.shape)


pc1 = sklearn_output[:,0]
pc2 = sklearn_output[:,1]
pc3 = sklearn_output[:,2]


g = sns.scatterplot(pc1, plot_cos, hue=labels)
g.set_xlabel('PC1')
g.set_ylabel('PC2')
plt.show()

outlier_dict = {}
for value,el,al in zip(pc1, extra_labels, labels):
    if value >  0.5:
        if al != "predicted":
            if el in outlier_dict:
                outlier_dict[el].append(al)
            else:
                outlier_dict[el] = [al]
    else:
        pass
        #print(value)

parse_pc = []
parse_cos = []
parse_labels = []
for value,el,al,cos in zip(pc1, extra_labels, labels, plot_cos):
    if el in outlier_dict:
       print(value)
       parse_pc.append(value)
       parse_cos.append(cos)
       parse_labels.append(al)
        

g = sns.scatterplot(parse_pc, parse_cos, hue=parse_labels)
g.set_xlabel('PC1')
g.set_ylabel('Cos High - Predicted')
plt.show()



with open('result.json', 'w') as fp:
    json.dump(outlier_dict, fp)
