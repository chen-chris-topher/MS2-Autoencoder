import requests
import json
import pandas as pd

#load the list of massive ids already used
with open('./msv_ids_in_use.txt', 'r') as f:
    used_msv = f.read().splitlines()    


gnps_datasets = requests.get("https://massive.ucsd.edu/ProteoSAFe/datasets_json.jsp").json()["datasets"]

instrument_variants = ['Q Exactive', 'Q Exactive Plus', 'Q Exactive HF', 'QE', 'QE Exactive', 'QExactive' \
'Thermo QExactive Orbitrap']

possible_datasets = []
for thing in gnps_datasets:
    msv_id = thing['dataset']
    if msv_id not in used_msv:
        instrument = thing['instrument']
        if instrument in instrument_variants:
            possible_datasets.append(msv_id)

dataset_dict = {'data':possible_datasets}
with open('./possible_dataset.json', 'w') as f:
    json.dump(dataset_dict, f)

