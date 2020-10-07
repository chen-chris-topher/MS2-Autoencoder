import os
import argparse
import concat_hdf5 as ch5

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='path to the data being concatenated')
parser.add_argument('data_name', help='name of the data file being concatenated')
parser.add_argument('--data_type', default='mz_array', help='mz array or additional information')
parser.add_argument('--name', default='big_data.hdf5', help='name of the resulting data file')
parser.add_argument('--norm', default='l2', help='the norm to use to normalize data')
parser.add_argument('--conv1d', default='False', help="do/don't reshape for Conv1D")

args = parser.parse_args()
path = args.data_path
data_name = args.data_name
name = args.name
norm = args.norm
conv1d = args.conv1d
data_type = args.data_type

#'/home/cchen/autoencode_output/'
#'C:/Users/CCheny/Documents/UC San Diego - Junction/Bioinformatics/MS2-Autoencoder/Output'

file_list = ch5.get_file_list(path, data_name)

if data_type == 'mz_array':
	ch5.stitch_hdf5(file_list, norm=norm, name=name) #data in autoencoder format
	#filename = name[:name.rfind('.')] + '_conv1d' + name[name.rfind('.'):]
	#ch5.stitch_hdf5_Conv1D(file_list, norm=norm, name=filename) #data in conv1d format
	print('operations complete')

else:
	ch5.wrangle_additional_data(file_list)


'''
if conv1d == 'False':
    ch5.stitch_hdf5(file_list, norm=norm, name=name)
    print('operations complete')

elif conv1d == 'True':
    filename = name[:name.rfind('.')] + '_conv1d' + name[name.rfind('.'):]
    ch5.stitch_hdf5_Conv1D(file_list, norm=norm, name=filename)
    print('operations complete')
'''
