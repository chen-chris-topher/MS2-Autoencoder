import os
import ast
import sys
import glob
import time
import requests
import zipfile
import pandas as pd
import numpy as np
import json

def merge_all_lib_files():
    all_merge_files = glob.glob('tsv_lib_results/*.tsv')
    first_pass = True
    lorge_list = []
    for count, test in enumerate(all_merge_files):
        print("Count", count)
        try:  
            df = pd.read_table(test, usecols = ['SpectrumID'])
        except:
            print("FAIL")
            continue
        df.drop_duplicates(inplace=True)
        comps = df['SpectrumID'].tolist()
        comps = list(np.unique(comps))
        
        lorge_list.extend(comps)
        lorge_list = list(np.unique(lorge_list))
        print("Lorge", len(lorge_list))
        
    json_string = json.dumps(lorge_list)
    with open('spectra_training_data.json','w') as json_file:
        json.dump(json_string, json_file)

def download_lib_search(tasks_full):
    url_1 = 'https://proteomics2.ucsd.edu/ProteoSAFe/DownloadResult?task='
    url_2 = '&view=view_all_annotations_DB'

    data = {"option": "delimit", "conten":"all"}

    tasks_full = np.unique(tasks_full).tolist()


    failed = [] 

    for task in tasks_full:
        try:
            os.system("rm %s_zip.zip" %task)
        except:
            pass

        try:
            os.sytem("rm MOLECULAR-LIBRARYSEARCH-V2-%s-view_all_annotations_DB-main.tsv" %task)
        
        except:
            pass

        if task == '':
            continue
        
        time.sleep(10)
        url = url_1 + task + url_2
        try:
            re = requests.post(url, allow_redirects=True, data = data) 
        
            with open("%s_zip.zip" %task, "wb") as code:
                code.write(re.content)
        
        except:
            print("%s task failed to download" %task)
            failed.append(task)
            continue
        time.sleep(20)    
        try:    
            with zipfile.ZipFile("%s_zip.zip" %task, 'r') as zip_ref:
                zip_ref.extractall(".")
        
        except:
            print("%s task failed to unzip" %task)
            failed.append(task)
            continue
        
        try:
            df = pd.read_tsv("MOLECULAR-LIBRARYSEARCH-V2-%s-view_all_annotations_DB-main.tsv" %task)
        
        except:
            print("Empty File %s" %task)
            failed.append(task)
            continue
        try:
            os.system('rm %s_zip.zip' %task)
        except:
            print("%s task failed to remove files" %task)

    print(failed)
def find_unique_compounds():
    """
    Written originally as an exploratory function, this now writes a 
    json of compounds to a .csv file.
    """

    with open('spectra_training_data.json', 'r') as json_file:
        data = json.load(json_file)
    data = ast.literal_eval(data)
    
    df = pd.DataFrame({"SpectrumID":data})
    df.to_csv('ccms_unique_lib_search_training.csv')

def get_task_ids():
    task_df = pd.read_table('global_tasks.tsv')
    task_list = task_df['taskid'].tolist()
    return(task_list)

def main():
    #task_list = get_task_ids()
    #download_lib_search(task_list)
    merge_all_lib_files()
    find_unique_compounds()

if __name__ == "__main__":
    main()
