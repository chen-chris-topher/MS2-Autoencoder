import os


def main():
    all_files = os.listdir('./spectra_data_4')
    for filename in all_files:
        full_name = os.path.join('./spectra_data_4', filename)
        if full_name.find('('):
            new_name = full_name.replace('(','_')
            new_name = new_name.replace(')','_')
            os.rename(full_name, new_name)

        elif full_name.find(')'):
            new_name = full_name.replace('(','_')
            os.rename(full_name, new_name)
        else:
            pass
if __name__ == "__main__":
    main()
