from pyteomics import mzxml, auxiliary
import numpy as np
import scipy

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
    """
    id_list_ms1 = []
    id_list_ms2 = []
    
    for i in range(0, len(data)): #looping over all scans in the file 
        if data[i]['polarity'] == '+': 
            if data[i].get('msLevel') == 2:
                id_list_ms2.append(i) #int 
            elif data[i].get('msLevel') == 1:
                id_list_ms1.append(i) #int
            else:
                print('msLevel error: could not sort dict[%s] msLevel' %(str(i)))

    #creating the files where this list will be stored
    filename1 = directory + '/id_list_ms1.txt'
    filename2 = directory + '/id_list_ms2.txt'
        
    #writing the data we just pulled out about the position/mslevel of the scans to a text file
    with open(filename1, 'w') as output:
        output.write(str(id_list_ms1)) #str in txt files for user review
    with open(filename2, 'w') as output:
        output.write(str(id_list_ms2)) #str in txt files for user review

    print('Generated list of indexes containing MS2 scans to "id_list_ms2.txt"')
    
    return id_list_ms2

def list_retentionTime_MS2(data, id_list_ms2):
    """
    list retentionTime for MS2 scans
    used to demonstrate syntax of accessing elements in the parsed mxzml
    """
    rt_list_ms2 = []

    for i in id_list_ms2:
        rt_list_ms2.append(data[i].get('retentionTime')) #syntax for accessing one particular key of one particular dictionary in the mzxml file

    return rt_list_ms2

def search_MS2_matches(data, id_list_ms2, rt_tol=0.5, mz_tol=0.01):
    """
    search for same molecules in MS2 scans
    same molecules are based on mass ('precursorMz') with a tolerance (mass_tolerance)
    same molecules are found within a bin of retentionTime (rt_tolerance)
    generates a dictionary of scan IDs mapped to a list of matching scan indexes
    outputs a .txt and .json file for review
    """
    print('Beginning the search')
    match_index_dict = {} #key is integer from id_list_m2; value is list of integers from id_list_ms2

    rt_tolerance = rt_tol #retentionTime tolerance for half a minute
    mass_tolerance = mz_tol #mass tolerance for 0.01mZ
    
    #WHY THESE PARAMTERS? DOES THIS MAKE SENSE, IS THIS WHY DATA GETS ZERO-ED?
    intensity_tolerance_low = 2 #greater than 2x precursorIntensity from base molecule
    intensity_tolerance_up = 5 #less than 5x precursorIntensity from base molecule

    
    for k in id_list_ms2: #looping over the scan numbers associated with MS2 scans
        rt_save = float(data[k].get('retentionTime')) #all of these look good for getting info out
        mass_save = float(data[k].get('precursorMz')[0].get('precursorMz'))
        intensity_save = float(data[k].get('precursorMz')[0].get('precursorIntensity'))
        id_save = int(data[k].get('id'))
        
        v_list = []
        redun_check = False #initialize redundancy check boolean as not-redundant

        #need to double check this, not enitrely sure why it's here
        for value in match_index_dict.values():
            for index in range(0, len(value)):
                if k == value[index]:
                    redun_check = True              

        if redun_check != True:
            for v in id_list_ms2: #looping through again looking for a high match
                rt_dv = data[v].get('retentionTime')
                
                if rt_dv <= rt_save + rt_tolerance and rt_dv >= rt_save - rt_tolerance:
                    
                    mass_dv = data[v].get('precursorMz')[0].get('precursorMz')
                    if mass_dv <= mass_save + mass_tolerance and mass_dv >= mass_save - mass_tolerance:
                    
                        intensity_dv = data[v].get('precursorMz')[0].get('precursorIntensity')
                        
                        #logic error here now fixed
                        if intensity_dv >= intensity_save:
                            v_list.append(v)
                            print('Found a match: %s:%r' %(k, v))
                
                match_index_dict[id_save] = v_list

            print('Finished search for dict[%s]' %k)
            redun_check = False #reset redundancy check boolean
        else:
            redun_check = False #reset redundancy check boolean 
    return match_index_dict

def get_match_scans(data, match_index_dict):
    """
    collect the information from the data for the matching molecules
    hierarchical dictionary
    """
    processed_dict = {}
    #loop through all the ms2 scans
    for key in match_index_dict.keys(): #key loops through scans
        
        processed_dict[int(key)] = []
        for index, i in zip(match_index_dict[key], range(0, len(match_index_dict[key]))): #where index loops through scans and i loops through samples
            scan = int(data[index].get('id'))
            rt = data[index].get('retentionTime')
            intensity = data[index].get('precursorMz')[0].get('precursorIntensity')
            mz = data[index].get('precursorMz')[0].get('precursorMz')
            mz_array = data[index].get('m/z array').tolist()
            intensity_array = data[index].get('intensity array').tolist()
            
            processed_dict[int(key)].append({scan:{}})
            processed_dict[int(key)][i][scan] = {'retentionTime':rt, #retentionTime
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
                binned_intensity, binned_mz, _ = binned_statistic(mz_array, intensity_array, statistic='sum', bins=200000, range=(0, 2000)) #bins are integers range(0,2000)
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
                binned_intensity, binned_mz, _ = binned_statistic(mz_array, intensity_array, statistic='sum', bins=200000, range=(0, 2000)) #bins are integers range(0,2000)
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
    print('successfully binned all intensity array')
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

def output_list(in_list, directory, two=None, ready_mass = None):
    import numpy as np
    if two == True:
        filename = directory + '/ready_array2.npz'
        np.savez_compressed(filename, in_list)
        print('saved ready_array2 to %s' %filename)
    elif ready_mass == True:
        filename = directory + '/ready_mass.npz'
        np.savez_compressed(filename, in_list)
        print('saved ready_array to %s' %filename)
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
