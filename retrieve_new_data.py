import os
import requests
import glob
import wget

from pathlib import Path
from ftplib import FTP
from multiprocessing import Pool


MSV_SERVER = 'massive.ucsd.edu'

def list_files(massive_id="MSV000086325"):
    os.system('wget -r -np -R "*.listing" -A "*.mzML" ftp://massive.ucsd.edu/%s -P' %massive_id)
    os.system('find ./massive.usd.edu -type d -empty -delete')

def main():
    considering_dataset = []

    filename = './massive_id_exclusion.txt'
    with open(filename) as f:
        content = f.readlines()
        content = [x.strip() for x in content] 

    gnps_datasets = requests.get("https://massive.ucsd.edu/ProteoSAFe/datasets_json.jsp").json()["datasets"]
    
    for thing in gnps_datasets:
        if 'QExactive' in thing['instrument']:
            pass
        elif 'Q Exactive' in thing['instrument']:
            pass
        else:
            continue

        if thing['dataset'] in content:
            continue
        else:
            pass

        considering_dataset.append(thing['dataset'])
    """
    pool = Pool(4)
    pool.map(untar_new, list_to_untar)
    pool.close()
    pool.join()
    """

if __name__ == "__main__":
    list_files()
