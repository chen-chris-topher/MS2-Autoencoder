import h5py
import numpy as np

def main():
    """
    with h5py.File("./big_data.hdf5", "r") as hf:
        df_low = hf["low_peaks"]
        print(df_low.shape)
        #print(type(df_low[0]))
    """


    file = np.load('./ready_array2.npz')
    data = file['arr_0']
    print(data.shape)

if __name__ == "__main__":
    main()
