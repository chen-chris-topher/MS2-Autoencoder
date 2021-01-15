import numpy as np
import json
import os
dir = './output_nf_2'
file = 'ordered_list2.json'
all_folders =os.listdir(dir)
all_files = [os.path.join(dir,item, file) for item in all_folders]
new_dict = {}
count = 0
for file in all_files:
    new_dict[file] = {}
    new_dict[file]['low'] = []
    new_dict[file]['high']=[]

for file in all_files:
    if os.path.exists(file):
        with open(file) as f:
            data = json.load(f) 
           
        for thing in data:
            if len(thing) != 0:
                for dict in thing:
                    count += 1                    
                    new_dict[file]['low'].append(dict[0])
                    new_dict[file]['high'].append(dict[1])
print(count)
with open('pairs.txt', 'w') as json_file:
    json.dump(new_dict, json_file) 
