import os
import argparse
import concat_hdf5 as ch5

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='path to the data being concatenated')
parser.add_argument('data_name', help='name of the data file being concatenated')
parser.add_argument('--name', default='big_data.hdf5', help='name of the resulting data file')
parser.add_argument('--norm', default='l2', help='the norm to use to normalize data')

args = parser.parse_args()
path = args.data_path
data_name = args.data_name
name = args.name
norm = args.norm

#'/home/cchen/autoencode_output/'
#'C:/Users/CCheny/Documents/UC San Diego - Junction/Bioinformatics/MS2-Autoencoder/Output'

file_list = ch5.get_file_list(path, data_name)
ch5.stitch_hdf5(file_list, norm=norm, name=name))
print('operations complete')