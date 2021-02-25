from pyteomics import mzxml, auxiliary
import numpy as np
import scipy
import os

def read_data(file):
    """
    read mzxml file using pyteomics.mzxml
    """

    data = mzxml.MzXML(file)
    print(str(file), 'has been accepted')

    return data

def count_MS2(data):
    """
    count total number scans and MS2 scans in the data
    """
    tot_ms2 = 0
    tot_ms1 = 0

    for i in range(0, len(data)):
        for k,v in data[i].items():
            if k == 'msLevel':
                if v == 2:
                    tot_ms2 += 1
                else:
                    tot_ms1 += 1
    print('Total %s scans in data' %(str(len(data))))
    print('Count %s MS2 scans in data' %(str(tot_ms2)))

def find_MS2(data, directory):
    """
    find MS2 scans from the data
    output to a list of indexes of MS2 scans
    with associated information on mz and rt for sorting
    """
    
    mass_tolerance = 0.01
    rt_tolerance = 0.5

    return_sort_dict = {}
    match_index_dict = {}

    #find and record all possible MS2 information
    for i in range(0, len(data)): #looping over all scans in the file 
        if data[i]['polarity'] == '+': 
            if data[i].get('msLevel') == 2: 
                
                inten = float(data[i].get('precursorMz')[0]['precursorIntensity'])
                mz = float(data[i].get('precursorMz')[0].get('precursorMz'))
                rt = float(data[i].get('retentionTime'))
                id = int(data[i].get('id'))      
 
                mz_array = data[i].get('m/z array').tolist()
                inten_array = data[i].get('intensity array').tolist()

                return_sort_dict[i] = {'mz': mz, 'rt': rt, 'inten' : inten, 'id':id, 'mz array': mz_array, 'inten array' : inten_array}
                
    #find all potential matches, regarless of precursor or overlap 
    for key, item in sorted(return_sort_dict.items(), key=lambda x: (x[1]['mz'], x[1]['rt']), reverse=True):
        mz1 = item['mz']
        rt1 = item['rt']
        id_save = item['id']
        prec1 = item['inten']

        two_list = []
        redun_check = False
       
        if redun_check is False:
            for key2, item2 in sorted(return_sort_dict.items(), key=lambda x: (x[1]['mz'], x[1]['rt']), reverse=True):
                #if key == key2:
                #    continue
                

                mz2 = item2['mz']
                rt2 = item2['rt']
                prec2 = item2['inten']
              
                if mz2 < mz1 - mass_tolerance:
                    break
                    
                #if we get below the target value by the mz tolerance, we're not getting higher
                if mz2 <= mz1 + mass_tolerance and mz2 >= mz1 - mass_tolerance:
                    if rt2 <= rt1 + rt_tolerance and rt2 >= rt1 - rt_tolerance:  
                        if prec2 >= prec1:           
                            two_list.append(key2)
                        
            if len(two_list) != 0:    
                match_index_dict[key] = two_list
            redun_check = False
            
            if key in match_index_dict.keys():
                print(key, match_index_dict[key])
            
            #print('%s of %s' %(key, len(return_sort_dict))) 
            #print('Finished search for dict[%s]' %key)
        else:
            redun_check = False
    return (match_index_dict, return_sort_dict)


def resolve_conflicts_pair(match_index_dict, return_sort_dict):
    
    for key, value in match_index_dict:
        pass 

def get_match_scans(sorted_dict, match_index_dict):
    """
    collect the information from the data for the matching molecules
    hierarchical dictionary
    """
    processed_dict = {}
    
    for key in match_index_dict.keys(): #key loops through scans   
        save_key = sorted_dict[key]['id']
        processed_dict[int(save_key)] = []
        save_key = sorted_dict[key]['id'] 
        
        for index, i in zip(match_index_dict[key], range(0, len(match_index_dict[key]))):
            scan = int(sorted_dict[index]['id'])
            rt = sorted_dict[index]['rt']
            intensity = sorted_dict[index]['inten']
            mz = sorted_dict[index]['mz']
            mz_array = sorted_dict[index]['mz array']
            intensity_array = sorted_dict[index]['inten array']
            
            processed_dict[int(save_key)].append({scan:{}})

            processed_dict[int(save_key)][i][scan] = {'retentionTime':rt, #retentionTime
                                                'precursorMz':mz, #precursorMz
                                                'precursorIntensity':intensity, #precursorIntensity
                                                'mz array':mz_array, #mz array 
                                                'intensity array':intensity_array} #intensity array
    return processed_dict

#use bin_array() for vecortizing and outputting zipped list of mz and intensity
def bin_array(processed_dict):
    """
    bin and zip mz array and intensity array
    mz values are binned
    intensity values are summed within the bin
    returns dictionary with binned and zipped array
    """
    from scipy.stats import binned_statistic

    binned_dict = {}
    for key in processed_dict.keys():
        binned_dict[key] = []
        for i in range(0, len(processed_dict[key])):
            for scan in processed_dict[key][i]:
                mz_array = processed_dict[key][i][scan].get('mz array')
                intensity_array = processed_dict[key][i][scan].get('intensity array')
                binned_intensity, binned_mz, _ = binned_statistic(mz_array, intensity_array, statistic='sum', bins=2000, range=(0, 2000)) #bins are integers range(0,2000)
                binned_mz = binned_mz[:-1]

                rt = processed_dict[key][i][scan].get('retentionTime')
                mz = processed_dict[key][i][scan].get('precursorMz')
                intensity = processed_dict[key][i][scan].get('precursorIntensity')
                #mz_intensity_array = np.dstack((binned_mz, binned_intensity)).reshape(len(binned_mz), 2) #zip binned mz array and binned intensity array
                mz_intensity_array = zip(binned_mz, binned_intensity)
                binned_dict[key].append({scan:{}})
                binned_dict[key][i][scan] = {'retentionTime':rt, #retentionTime
                                            'precursorMz':mz, #precursorMz
                                            'precursorIntensity':intensity, #precursorIntensity
                                            'mz_intensity array':mz_intensity_array} #intensity array
    print('successfully binned all mz array and intensity array')
    return binned_dict

#use bin_array2() for vectorizing and outputting only the intensity array
def bin_array2(processed_dict):
    """
    bin intensity array
    mz values are binned
    intensity values are summed within the bin
    returns dictionary with binned intensity array
    """
    from scipy.stats import binned_statistic

    binned_dict = {}
    for key in processed_dict.keys():
        binned_dict[key] = []
        for i in range(0, len(processed_dict[key])):
            for scan in processed_dict[key][i]:
                mz_array = processed_dict[key][i][scan].get('mz array')
                intensity_array = processed_dict[key][i][scan].get('intensity array')
                binned_intensity, binned_mz, _ = binned_statistic(mz_array, intensity_array, statistic='sum', bins=2000, range=(0, 2000)) #bins are integers range(0,2000)
                binned_mz = binned_mz[:-1]

                rt = processed_dict[key][i][scan].get('retentionTime')
                mz = processed_dict[key][i][scan].get('precursorMz')
                intensity = processed_dict[key][i][scan].get('precursorIntensity')
                intensity_array = binned_intensity.tolist()
                binned_dict[key].append({scan:{}})
                binned_dict[key][i][scan] = {'retentionTime':rt, #retentionTime
                                            'precursorMz':mz, #precursorMz
                                            'precursorIntensity':intensity, #precursorIntensity
                                            'intensity array':intensity_array} #intensity array
    return binned_dict

def create_pairs(binned_dict):
    """
    creates pairs of scans from dict of matched scans
    number of pairs per same molecule is n(n+1)/2 where n is number of scans
    returns list with paired scans
    """
    pairs_list = []
    for key in binned_dict.keys(): #looping through all binned MS2 scans
        pairs = []
        for i in range(0, len(binned_dict[key])):
            for j in range(i+1, len(binned_dict[key])): 
                for scan, scan2 in zip(binned_dict[key][i].keys(), binned_dict[key][j].keys()):
                    if np.count_nonzero(scan) != 0:
                        if np.count_nonzero(scan2):
                            pairs.append([binned_dict[key][i][scan], binned_dict[key][j][scan2]])
                            
        pairs_list.append(pairs)
    print('successfully created pairs for all matched scans')
    return pairs_list

def arrange_min_max(pairs_list):
    """
    rearrange each match pair so that the smaller precursorIntensity is first
    and the bigger precursorIntensity is second
    input is a list
    returns list with arranged pairs
    """
    ordered_list = []
    for i in range(0, len(pairs_list)): #i is at the group/molecule level
        pairs = []
        for j in range(0, len(pairs_list[i])): #j is at the pairs per molecule level
            if pairs_list[i][j][0].get('precursorIntensity') <= pairs_list[i][j][1].get('precursorIntensity'):
                pairs.append([pairs_list[i][j][0], pairs_list[i][j][1]])
            elif pairs_list[i][j][1].get('precursorIntensity') < pairs_list[i][j][0].get('precursorIntensity'):
                pairs.append([pairs_list[i][j][1], pairs_list[i][j][0]])
        ordered_list.append(pairs)
    len_pairs = len(pairs_list)
    len_ordered = len(ordered_list)

    print('length of ordered list %s should be the same as length of pairs list %d' % (len_ordered, len_pairs))
    return ordered_list

#use convert_to_ready() for creating an all inclusive file
def convert_to_ready(ordered_list):
    """
    converts ordered_list into a list of structured arrays
    without dictionary keys
    conversion creates a list with all useful information
    """
    ready_list = []
    ready_list_2 = []
    ready_array = [] #just needs declaration
    ready_mz = [] #just needs declaration
    
    for i in range(0, len(ordered_list)): #i is at the group/molecule level
        group = []
        for j in range(0, len(ordered_list[i])): #j is at the pairs per molecule level
            pairs = []
            mz_intensities =[]
            for k in range(0 , len(ordered_list[i][j])): #k is at the scan per pair level
                rt = ordered_list[i][j][k].get('retentionTime')
                intensity = ordered_list[i][j][k].get('precursorIntensity')
                mz = ordered_list[i][j][k].get('precursorMz')
                mz_intensity_array = np.asarray(ordered_list[i][j][k].get('mz_intensity array'))
                
                mz_intensities.append(mz_intensity_array)
                pairs.append(np.asarray([rt, intensity, mz]))
        
            ready_list.append(np.asarray(pairs))
            ready_list_2.append(np.asarray(mz_intensities))

        ready_mz=np.asarray(ready_list_2)
        ready_array = np.asarray(ready_list)
    
    return(ready_array, ready_mz)

#use convert_to_ready2() for creating a training ready file
def convert_to_ready2(ordered_list):
    """
    converts ordered_list into a list of structured arrays
    without dictionary keys
    renders only the mz_intensity array 
    conversion makes list ready as training input
    """
    ready_list = []
    ready_array = []
    for i in range(0, len(ordered_list)): #i is at the group/molecule level
        for j in range(0, len(ordered_list[i])): #j is at the pairs per molecule level
            pairs = []
            for k in range(0 , len(ordered_list[i][j])): #k is at the scan per pair level
                intensity_array = ordered_list[i][j][k].get('intensity array')
                pairs.append(np.asarray(intensity_array))
            ready_list.append(np.asarray(pairs))
        ready_array = np.asarray(ready_list)
    return ready_array

def convert_to_ready3(ordered_list):
    """
    converts ordered_list into a list of structured DENSE arrays
    without dictionary keys
    renders only the DENSE mz_intensity array 
    conversion makes list ready as training input
    """
    ready_list = []

    for i in range(0, len(ordered_list)): #i is at the group/molecule level
        for j in range(0, len(ordered_list[i])): #j is at the pairs per molecule level
            pairs = []
            for k in range(0 , len(ordered_list[i][j])): #k is at the scan per pair level
                intensity_array = ordered_list[i][j][k].get('intensity array')
                pairs.append(list(intensity_array))

            ready_list.append(list(pairs))
        rows, cols, vals = zip(*ready_list)
        a = scipy.sparse.coo_matrix((vals, (rows, cols)))
        ready_array = a.A
    ready_array.todense()
    return ready_array


def output_file(in_dict, directory, match_index=None, processed=None, binned=None, pairs=None, ordered=None):    
    """
    output the dictionary from search_MS2_matches, get_match_scans, bin_array, create_pairs, arrange_min_max into files
    outputs .json
    """
    import json

    if match_index == True:
        json = json.dumps(in_dict)
        filename = directory + '/match_index.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved match_index_dict to %s' %filename)

    elif processed == True:
        json = json.dumps(in_dict)
        filename = directory + '/processed_dict.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved processed_dict to %s' %filename)

    elif binned == True:
        json = json.dumps(in_dict)
        filename = directory + '/binned_dict.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved binned_dict to %s' %filename)
    
    elif pairs == True:
        json = json.dumps(in_dict)
        filename = directory + '/pairs_list.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved pairs_list to %s' %filename)
    
    elif ordered == True:
        json = json.dumps(in_dict)
        filename = directory + '/ordered_list.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved ordered_list to %s' %filename)

    else:
        json = json.dumps(in_dict)
        filename = directory + '/output.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved dict to "output.json"')

def output_file2(in_dict, directory, binned=None, pairs=None, ordered=None):    
    """
    output the dictionary from bin_array2, create_pairs, arrange_min_max into files
    outputs .json
    """
    import json

    if binned == True:
        json = json.dumps(in_dict)
        filename = directory + '/binned_dict2.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved binned_dict to %s' %filename)
    
    elif pairs == True:
        json = json.dumps(in_dict)
        filename = directory + '/pairs_list2.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved pairs_list to %s' %filename)
    
    elif ordered == True:
        json = json.dumps(in_dict)
        filename = directory + '/ordered_list2.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved ordered_list to %s' %filename)

    else:
        json = json.dumps(in_dict)
        filename = directory + '/output.json'
        with open(filename, 'w') as output:
            output.write(json)
        print('saved dict to "output.json"')

def output_list(in_list, directory, two=None, ready_mass = None, dict=False):
    import numpy as np
    import json
    if two == True:
        filename = directory + '/ready_array2.npz'
        np.savez_compressed(filename, in_list)
        print('saved ready_array2 to %s' %filename)
    elif ready_mass == True:
        filename = directory + '/ready_mass.npz'
        np.savez_compressed(filename, in_list)
        print('saved ready_array to %s' %filename)
    elif dict == True: 
        json = json.dumps(in_list)
        filename = directory + '/ready_array_dict.npz'
        with open(filename, 'w') as output:
            output.write(json)
    else:
        filename = directory + '/ready_array.npz'
        np.savez_compressed(filename, in_list)
        print('saved ready_array to %s' %filename)

def unpack(input_dict):
    """
    unpack a dictionary that has be save in a .json file
    """
    import json
    with open(input_dict) as f:
        out_dict = json.load(f)
    return out_dict
