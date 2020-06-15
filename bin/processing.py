import os
import argparse
import glob
import concat_hdf5 as ch5

parser = argparse.ArgumentParser()
parser.add_argument('data_path', help='path to the data being concatenated')
parser.add_argument('data_name', help='name of the data file being concatenated, likely to be ready_array2.npz')
parser.add_argument('--name', default='big_data.hdf5', help='name of the resulting data file')
parser.add_argument('--norm', default='l2', help='the norm to use to normalize data')
parser.add_argument('--conv1d', default='False', help="do/don't reshape for Conv1D")

args = parser.parse_args()
path = args.data_path
data_name = args.data_name
name = args.name
norm = args.norm
conv1d = args.conv1d

file_list = glob.glob(os.path.join(args.data_path, "**/{}".format(data_name)), recursive=True)

ch5.stitch_hdf5(file_list, norm=norm, name=name) #data in autoencoder format
#filename = name[:name.rfind('.')] + '_conv1d' + name[name.rfind('.'):]
#ch5.stitch_hdf5_Conv1D(file_list, norm=norm, name=filename) #data in conv1d format
print('operations complete')

'''
if conv1d == 'False':
    ch5.stitch_hdf5(file_list, norm=norm, name=name)
    print('operations complete')

elif conv1d == 'True':
    filename = name[:name.rfind('.')] + '_conv1d' + name[name.rfind('.'):]
    ch5.stitch_hdf5_Conv1D(file_list, norm=norm, name=filename)
    print('operations complete')
'''
