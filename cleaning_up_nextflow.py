import os
import sys


def main():
	spectra_location = './spectra_data'
	output_location = './output_nf_2'

	all_spectra_downloaded = os.listdir(spectra_location)
	possible_output = os.listdir(output_location)


	possible_output = [os.path.join(output_location, item, 'ready_array2.npz') for item in possible_output]
	complete_output = []
	for output_file in possible_output:
		if os.path.isfile(output_file):
			complete_output.append(output_file)
	
	spectra_removal_names = [item.split('/')[2].replace("_outdir","") + '.mzXML' for item in complete_output]

	for spectra in spectra_removal_names:
		try:
			os.remove(os.path.join(spectra_location, spectra))
			print(os.path.join(spectra_location,spectra))
		except:
			print(os.path.join(spectra_location,spectra), ' Not Removed')


if __name__ == "__main__":
	main()
