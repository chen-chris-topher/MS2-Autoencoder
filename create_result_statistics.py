import os
import sys
import matplotlib.pyplot as plt
import spectrum_utils.plot as sup
import spectrum_utils.spectrum as sus
from scipy import spatial



#input all actual versus predicted spectra, output cos score list
def cosine_scores(predicted_spectra, actual_spectra):
	cosine_scores = []

	for predict_vector, real_vector in zip(predicted_spectra, actual_spectra):
		#calcualtes cosine similarity
		result = 1 - spatial.distance.cosine(predict_vector, real_vector)
		cosine_scores.append(result)
	
	return(cosine_scores)

#input all actual and predicted spectra and numbe of mirror plots to be created
def create_mirror_plots(predicted_spectra, actual_spectra, num=10):
	while count <= num:
		







		count += 1

