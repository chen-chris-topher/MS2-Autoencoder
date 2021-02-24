import sys
import h5py
import numpy as np
import json
def main():
    """
    with h5py.File("./big_data.hdf5", "r") as hf:
        df_low = hf["low_peaks"]
        print(df_low.shape)
        #print(type(df_low[0]))
    """
    """ 
    file = np.load('./ready_array2.npz')
    #file = np.load('../output_nf_1/MSV000079034_raw_150116_EF_PMA_PSN474_LN-3442-f_ddMS2_pos_outdir/ready_array2.npz')
    data = file['arr_0']
    print(data.shape) 
    #final_count = 0
    #for count,thing in enumerate(data):
    #    if np.count_nonzero(thing) > 0:
    #        final_count += 1
    #print(final_count)
    sys.exit(0) 
    """
    filename1 = './processed_dict_og.json'
    filename2 = './processed_dict_optimized.json'


    #filename1 = './output.json'
    #filename2 = './output_optimized.json' #optmizied keys are 1 behind the ids

    with open(filename1, 'r') as hf:
        data1 = json.load(hf)
    with open(filename2, 'r') as hf:
        data2 = json.load(hf)
 
    keys2 = list(data2.keys())
    keys1 = list(data1.keys())
    #print(keys2)
    #print(keys1)  
 
    keys2.sort() 
    keys1.sort()
 
    #print(keys1)

    #keys2 = [str(int(item) +1) for item in keys2]
    #print(keys2)
    
    #these are things that are in the optimized but not regular
    print([item for item in keys2 if item not in keys1])
    
    #these are in the regular but not the optimized
    print([item for item in keys1 if item not in keys2])

    #print(len(keys2))
    #print(len(keys1))
    sys.exit(0)
    for key, value in data1.items():
        #value = [item+1 for item in value]
       
        try:
            value2 = data2[key]
        except:
            print("key not found")
            print(key, value)
            continue
        if key == '':
            print('bad key')
        if value != value2:
            print("values dont match")
            print(value, value2, key)
            


if __name__ == "__main__":
    main()
