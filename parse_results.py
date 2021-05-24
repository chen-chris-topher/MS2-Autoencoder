import os
import h5py
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.spatial.distance import cosine

def locate_all_files_dict():
    """
    Makes an in order list of while files have
    which percetn of noise added to them.
    """

    #first we find the order of ccms spectra
    ccms_list = []
    with open('./ccms_msreduce_og.txt') as hf:
        for line in hf:
            ccms_list.append(line.strip())
    
    ccms_list = ccms_list[:10000] * 20
    
    unique_ccms_list = np.unique(ccms_list)
    
    target_files = []
    for ccms in unique_ccms_list:
        second_part = ccms.split('.')[0][-6:]
        target_files.append(ccms + '.' + second_part + '.0.dta')

    base_count = {}    
    exclusion_ccms = []
    for target in target_files:
        target = os.path.join('dta_files', target)
        count = 0
        if os.path.isfile(target):
            with open(target, 'r') as tf:
                for line in tf:
                    count += 1
            base_count[target] = count - 1
        else:
            exclusion_ccms.append(target)
        
    filenames = os.listdir('dta_files')
    open_filenames = [os.path.join('dta_files',item) for item in filenames]

    all_file_count = {}
    for all_file in open_filenames:
        count = 0
        with open(all_file, 'r') as af:
            for line in af:
                count += 1
        all_file_count[all_file] = count - 1
    
    #key = filename, value = percentage of noise
    percent_noise_dict = {}

    for filename, peaks in all_file_count.items():
        break_point = filename.split('.')[:-2]
        break_point = '.'.join(break_point) + '.0.dta'
        if break_point not in base_count:
            continue
        normal_peaks = base_count[break_point]
        
        #contains percent of peaks to keep by filenames         
        all_file_count[filename] = (normal_peaks / peaks) * 100

    return(all_file_count)

def run_ms_reduce(percent_data_keep):
    partition_data = range(10, 100, 10)
    
    vals = list(percent_data_keep.values())
    sns.distplot(vals)
    plt.show()
    total_files = []
    #run for each percent 
    for percent in partition_data:
        print(percent)
        files_to_run = []
        
        for key, value in percent_data_keep.items():

            if value > (percent - 5) and value <= (percent + 5):
                files_to_run.append(key)
        total_files.extend(files_to_run) 
        
        #s.system('rm test_file/*')
        #for filename in files_to_run:
        #    os.system('cp %s test_file/' %filename)
        
        #os.system('javac msreduce.java')
        #os.system('java msreduce test_file/ %s output_dir/' %percent)
    with open('used_files.txt', 'w') as hf:
        for f in total_files:
            hf.write(f + '\n')

def record_msreduce_order():
    import time
    all_results = os.listdir('output_dir/')
    all_results = [os.path.join('output_dir', item) for item in all_results]

    all_dta_files = [os.path.join('dta_files', item) for item in os.listdir('dta_files')]
    
    order = []
    total_count = 0
    start = time.time()
    for filename in all_results:
        print(filename)
        
        with open(filename, 'r') as hf:
            for line in hf:
                if line.startswith('S'):
                    total_count += 1
                    
    print(total_count)

def analyze_msreduce_results():
    import time
    all_results = os.listdir('output_dir/')
    all_results = [os.path.join('output_dir', item) for item in all_results]

    all_dta_files = [os.path.join('dta_files', item) for item in os.listdir('dta_files')]
    
    ms_reduce_matrix = np.zeros((200000,2000))
    og_matrix = np.zeros((200000, 2000))
    match = False

    order = []

    matched_file_list = []
    total_count = 0
    start = time.time()
    for filename in all_results:
        print(filename)
        with open(filename, 'r') as hf:
            for line in hf:
                if line.startswith('S'):
                    total_count += 1
                    if total_count % 1000 == 0:
                        end = time.time()
                        print(end - start)
                        print(total_count)
                    if match is True:
                        matched_file_list.append(dta_file_match) 
                        msr_binned_intensity, binned_mz, _ = binned_statistic(ms_reduce_peaks, ms_reduce_intensity, statistic='sum', bins=2000, range=(0, 2000))
                        og_binned_intensity, og_binned_mz, _ = binned_statistic(og_peaks, og_intensity, statistic='sum', bins=2000, range=(0, 2000))
                        og_matrix[total_count-1] = og_binned_intensity
                        ms_reduce_matrix[total_count-1] = msr_binned_intensity
                        
                    add_peak = line.strip().split('\t')[1]
                    ms_reduce_peaks = []
                    ms_reduce_intensity = []
                    og_peaks = []
                    og_intensity = []
                    match = False

                elif line.startswith('Z'):
                    prec, mz  = line.strip().split('\t')[1:]
                    
                    for dta_file in all_dta_files:
                        dta_add_peak = dta_file.split('.')[-2]
                        if dta_add_peak != add_peak:
                            continue
                       
                        #if this file matches the peak number
                        with open(dta_file, 'r') as dta:
                            match = False
                            for line_x in dta:
                                dta_mz, dta_prec = line_x.strip().split(" ")     
                                
                                #if this file is an exact match
                                if dta_mz == mz and dta_prec == prec:
                                    match = True  
                                    dta_file_match = dta_file
                                    order.append(dta_file)
                                    continue
                                if match is True:
                                    og_intensity.append(float(line_x.strip().split(" ")[1]))
                                    og_peaks.append(float(line_x.strip().split(" ")[0]))
                        
                        if len(og_peaks) > 0:
                            break
                        
                #if we made it to this pont we have a match
                elif match is True:
                    ms_reduce_peaks.append(float(line.strip().split('\t')[0]))
                    ms_reduce_intensity.append(float(line.strip().split('\t')[1]))
                
                elif match is False:
                    print(line.strip())
                    print(add_peak)
                    print(prec)
                    print(mz)

    #print([item for item in all_dta_files if item not in matched_file_list])
    with open('order.txt','w') as d:
        for line in order:
            d.write(line + '\n')


    print(ms_reduce_matrix.shape)
    np.save('./ms_reduce.npy', ms_reduce_matrix)
    print(og_matrix.shape)
    np.save('./original_peaks.npy', og_matrix)
    print(total_count)


def eval_cosine():
    msr = np.load('ms_reduce.npy')
    og = np.load('original_peaks.npy')
    print(msr.shape)
    print(og.shape)
    all_cos = []
    for mr, o in zip(msr, og):
        if np.max(mr) > 0 and np.max(o) > 0:
            all_cos.append(1-cosine(o,mr))
    print(len(all_cos))
    sns.distplot(all_cos)
    plt.show()

def eval_percent_noise():
    #just to be clear, original_peaks is noisy peaks

    file_order = []
    with open('order.txt', 'r') as od:
        for line in od:
            filename = line.strip() 
            file_list = filename.split('.')
            file_list[2] = '0'
            filename = '.'.join(file_list)
            file_order.append(filename)
    
    matrix = np.zeros((200000, 2000))
    for i,file_name in enumerate(file_order):
        if i % 1000 == 0:
            print(i)
        with open(file_name, 'r') as fn:
            mass = []
            inten = []
            for count,line in enumerate(fn):
                if count == 0:
                    continue
                else:
                    line = line.strip()
                    mass.append(float(line.split(' ')[0]))
                    inten.append(float(line.split(' ')[1]))
                    
            binned_intensity, binned_mz, _ = binned_statistic(mass, inten, statistic='sum', bins=2000, range=(0, 2000))
            matrix[i] = binned_intensity
    
    np.save('./ms_reduce_noiseless.npy',matrix)
def zip_npy_matrix():
    msr = np.load('ms_reduce.npy')[1:]
    noisy = np.load('original_peaks.npy')[1:]
    og = np.load('./ms_reduce_noiseless.npy')

    missing_noisy = np.load('./missing_values_noise_added.npy')
    missing_og = np.load('missing_values_original.npy')
    missing_msr = np.zeros((78526,2000))
    
    order = []
    with open('order.txt','r') as d:
        for line in d:
            order.append(line.strip())
    print(order[1])
    print(missing_noisy.shape)
    for count, (a,b,c) in enumerate(zip(msr,noisy,og)):
        if np.max(a) == 0 and np.max(b) == 0 and np.max(c) == 0:
            print(count)
            break      

    msr = msr[~np.all(msr == 0, axis=1)]
    print(msr.shape)
    new_msr = np.concatenate((msr,missing_msr), axis =0)
   
    og = og[~np.all(og == 0, axis=1)]
    print(og.shape)
    new_og = np.concatenate((og, missing_og), axis=0) 

    noisy = noisy[~np.all(noisy == 0, axis=1)]
    print(noisy.shape)
    new_noisy = np.concatenate((noisy, missing_noisy), axis=0)
    

    np.save('full_msr.npy', new_msr)
    np.save('full_og.npy', new_og)
    np.save('full_noisy.npy', new_noisy)

def make_figure():
    msr = np.load('ms_reduce.npy')
    noisy = np.load('original_peaks.npy')
    og = np.load('./ms_reduce_noiseless.npy')
    
    
    missing_noisy = np.load('./missing_values_noise_added.npy')
    missing_og = np.load('missing_values_original.npy')
    missing_msr = np.zeros((78526,2000))
    

    file_order = []
    with open('order.txt', 'r') as od:
        for line in od:
            filename = line.strip() 
            #ile_list = filename.split('.')
            #ile_list[2] = '0'
            #ilename = '.'.join(file_list)
            file_order.append(filename)
    #print(file_order[50829])
    #sys.exit(0)
    print(msr.shape)
    print(noisy.shape)
    print(og.shape)
    print(len(file_order)) 

    og_noisy = []
    reduced_noisy = []
    count = 0
    label = []
    val = []

    for a,b in zip(msr, noisy):
        if np.max(a) >0 and np.max(b) > 0:
            count += 1
     
    #msr is predcted, noisy is low, of is high
    for count, (a,b,c) in enumerate(zip(msr[1:], noisy[1:], og)):    
        a_h = np.amax(a)
        if a_h == 0:
            continue
        og_noisy.append(1-cosine(b,c))
        reduced_noisy.append(1-cosine(c,a))
        
        pre_high_cos= 1-cosine(b,a)
        low_high_cos = 1-cosine(b,c)
        val.append(pre_high_cos-low_high_cos)
        if low_high_cos >= 0.80 and low_high_cos < 0.85:
            label.append('0.80-0.85')
        elif low_high_cos >=0.85 and low_high_cos < 0.90:
            label.append('0.85-0.90')
        elif low_high_cos >= 0.90 and low_high_cos < 0.95:
            label.append('0.90-0.95')
        elif low_high_cos >= 0.95:
            label.append('0.95-1.0')
        else:
            continue


        
    for count, (a,b,c) in enumerate(zip(missing_msr, missing_noisy, missing_og)):
        
        #print(1-cosine(b,a))
        og_noisy.append(1-cosine(b,c))
        reduced_noisy.append(1-cosine(c,a))
        pre_high_cos= 1-cosine(b,a)
        low_high_cos = 1-cosine(b,c)
        val.append(pre_high_cos-low_high_cos)
        if low_high_cos >= 0.80 and low_high_cos < 0.85:
            label.append('0.80-0.85')
        elif low_high_cos >=0.85 and low_high_cos < 0.90:
            label.append('0.85-0.90')
        elif low_high_cos >= 0.90 and low_high_cos < 0.95:
            label.append('0.90-0.95')
        elif low_high_cos >= 0.95:
            label.append('0.95-1.0')
        else:
            continue
    
    """
    final_lib = np.load('./original.npy')
    final_lib = bin_reduce_normalize(final_lib)
    noisy_peaks = np.load('./noise_added.npy')
    noisy_peaks = bin_reduce_normalize(noisy_peaks)
    denoised_data = np.load('../MS2-Autoencoder/benchmark/predictions.npy') 
	"""

    print(len(val))
    print(len(label))
    ax = sns.boxplot(x=label, y=val, order=["0.80-0.85", "0.85-0.90","0.90-0.95", "0.95-1.0"], palette="Purples")
    plt.axhline(y=0, color='orange')
    plt.show()
    plt.close()

    fig = plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    ax = sns.distplot(reduced_noisy, color = 'orange', hist=False)
    sns.distplot(og_noisy, color = 'purple', hist=False)
    #sns.distplot(other_cos, color='blue', hist=False)
    ax.set_xlim(0.5, 1)
    fig.legend(labels=['MSREDUCE Predicted vs. Original','Noisy vs. Original'])
    plt.show()

    plt.close()
    sys.exit(0) 
    percent_of_noise_removed = []
    percent_of_noise_not_removed = []
    fpr = []
    #missing_msr, missing_noisy, missing_og
    fin_msr = np.concatenate((msr,missing_msr))
    fin_noisy = np.concatenate((noisy, missing_noisy))
    fin_og = np.concatenate((og, missing_og))
    
    print(fin_msr.shape)
    print(fin_noisy.shape)
    print(fin_og.shape)

    for a,b,c in zip(fin_msr[1:], fin_noisy[1:], fin_og):  
        a_h = np.amax(a)
        if a_h == 0:
            continue
        noise_kept = 0
        noise_removed = 0
        noise_added  = 0
        false_peak_removal = 0
        total_real_peaks = 0
        for mz1, mz2, mz3 in zip(a,b,c):
            if mz3 > 0 and mz1 == 0:
                false_peak_removal += 1
            if mz3 > 0:
                total_real_peaks += 1
             
            #it's predicted, in the noisy but not in the original
            if mz1 > 0 and mz2 > 0 and  mz3 == 0:
                noise_kept += 1
            #if its in the noisy but not the original    
            if mz2 > 0 and mz3 == 0:
                noise_added += 1

            #not in the predicted, not in the original, in the noisy
            if mz2 > 0 and mz1 == 0 and mz3 == 0:    
                noise_removed += 1
            
        if total_real_peaks > 0:
            fpr.append(false_peak_removal/total_real_peaks)
        
        if noise_added == 0:
            continue
        if noise_removed / noise_added > 1:
            print(noise_removed, noise_added, noise_kept)
        percent_of_noise_not_removed.append(noise_kept / noise_added) #percent of remaining noise
        percent_of_noise_removed.append(noise_removed / noise_added)
        
        
    fig = plt.figure(figsize=(10,6))
    sns.set_style("whitegrid")
    sns.distplot(fpr, hist=False, color = 'blue')
    plt.show()
    plt.close()

    ax = sns.distplot(x=percent_of_noise_removed, color = 'blue', hist = False)    
    #sns.distplot(x=percent_of_noise_not_removed, color= 'darkgreen', hist=False)
    fig.legend(labels=['Percent of Noise Removed'])
    plt.xlabel('Percent of Noise Removed')
    plt.ylabel('Distribution')
    plt.show()

def missing_file():
    used = []
    miss_cos = []

    msr = np.load('./ms_reduce.npy')
    

    with open('order.txt','r') as hf:
        for line in hf:
            used.append(line.strip())
   
    shoulda_used = []
    with open('used_files.txt', 'r') as hf:
        for line in hf:
            shoulda_used.append(line.strip())
            
    #issing = [item for item in shoulda_used if item not in used]
  
    missing = [os.path.join('dta_files',item) for item in os.listdir('dta_files')]
    missing = [item for item in missing if item.split('.')[-2] != '0']
    missing_matrix = np.zeros((190000,2000))
    og_missing_matrix = np.zeros((190000,2000))
    msr_matrix = np.zeros((190000, 2000))

    msr_county = 0
    nccms = []
    vec =[]
    for count,item in enumerate(missing):
        mass = []
        inten = []
        
        original_item = item.split('.')
        original_item[-2] = '0'
        ccms_n = original_item[0].split('/')[1]
        original_item = '.'.join(original_item)
        if item == original_item:
            continue
        
        try:
            msr_loc = used.index(item)
            vec = msr[msr_loc+1]
            msr_matrix[count] = msr[msr_loc+1]
            msr_county += 1
            
        except: 
            pass
 
        nccms.append(item)
        #print(item, original_item) 
        with open(item, 'r') as hf:
            for i, line in enumerate(hf):
                if i > 0:
                  m = line.strip().split(' ')[0]
                  i = line.strip().split(' ')[1]
                  mass.append(float(m))
                  inten.append(round(float(i),4))
        binned_intensity, binned_mz, _ = binned_statistic(mass, inten, statistic='sum', bins=2000, range=(0, 2000))
        missing_matrix[count] = binned_intensity 

        mass_o = []
        inten_o  = []
        with open(original_item, 'r') as hf:
            for i, line in enumerate(hf):
                if i > 0:
                  m = line.strip().split(' ')[0]
                  i = line.strip().split(' ')[1]
                  mass_o.append(float(m))
                  inten_o.append(round(float(i),4))
        
        binned_intensity_o, binned_mz, _ = binned_statistic(mass_o, inten_o, statistic='sum', bins=2000, range=(0, 2000))
        print(np.max(binned_intensity_o), item)
        og_missing_matrix[count] = binned_intensity_o
        if count % 1000 == 0:
            print(count)
    
    #print(miss_cos[:10])
    #sns.distplot(miss_cos)
    #plt.show()
    print(msr_county) 
    np.save('original.npy', og_missing_matrix)
    np.save('noise_added.npy', missing_matrix)
    np.save('msr_final.npy', msr_matrix)
    with open('new_ccms_list.txt', 'w') as hf:
        for line in nccms:
            hf.write(line + '\n')


def main():
    #zip_npy_matrix()
    #missing_file()
    #percent_data_keep = locate_all_files_dict()
    #print(percent_data_keep)
    #run_ms_reduce(percent_data_keep)
    #record_msreduce_order()
    #nalyze_msreduce_results()
    #eval_percent_noise()
    #val_cosine()
    make_figure()

if __name__ == "__main__":
    main()
