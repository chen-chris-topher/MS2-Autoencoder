import h5py


def main():
    with h5py.File("../big_data.hdf5", "r") as hf:
        df_low = hf["low_peaks"]
        print(df_low.shape)
        print(type(df_low[0]))


if __name__ == "__main__":
    main()
