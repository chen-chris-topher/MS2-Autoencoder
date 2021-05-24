import os
import sys
import json
import pandas as pd

reg_table = 'lib_1.tsv'
predict_table = 'pre_1.tsv'
noisy_table = 'noi_1.tsv'
name_table = 'compound_names_1_10000.csv'

def main():
    reg = pd.read_table(reg_table)
    pre = pd.read_table(predict_table)
    noi = pd.read_table(noisy_table)
    name = pd.read_csv(name_table)
    print(name)
    
    """
    print(pre)
    print(noi)
    print(reg)
    """
   
    col_list = ['#Scan#', 'Compound_Name', 'SpectrumID', 'INCHI', 'Smiles', 'MQScore', 'SharedPeaks']
    reg = reg[col_list]
    pre = pre[col_list]
    noi = noi[col_list]

    reg.columns = [str(col) + '_library' for col in reg.columns]
    pre.columns = [str(col) + '_predictions' for col in pre.columns]
    noi.columns = [str(col) + '_noisy' for col in noi.columns]

    ccms_list = []
    with open('new_ccms_list.txt') as hf:
        for line in hf:
            ccms_list.append(line.strip())
    #ccms_list_add = ccms_list[:10000]
    #ccms_list.extend(ccms_list_add)
    print(len(ccms_list))
    
     
    ccms_df = pd.DataFrame({'#Scan#_ccms': range(1, 190001), 'SpectrumID_ccms':ccms_list})
    merge_0 = ccms_df.merge(name, on = 'SpectrumID_ccms', how = 'left')
    merge_1 = merge_0.merge(pre, left_on = '#Scan#_ccms', right_on = '#Scan#_predictions', how = 'outer')
    merge_2 = merge_1.merge(noi, left_on = '#Scan#_ccms', right_on = '#Scan#_noisy', how = 'outer')
    merge_3 = merge_2.merge(reg, left_on = '#Scan#_ccms', right_on = '#Scan#_library', how = 'outer')

    
    merge_3 = merge_3.astype(str).apply(lambda x: x.str.lower())
    merge_3.replace({'"': '', 'inchi=': ''},regex=True, inplace=True) 
    
    print(merge_3)
    match_pre = []
    match_noisy = []
    match_lib = []

    for index, row in merge_3.iterrows():
        ccms_id = str(row['SpectrumID_ccms'])
        lib_id = str(row['SpectrumID_library'])
        pre_id = str(row['SpectrumID_predictions'])
        noi_id = str(row['SpectrumID_noisy'])
       
        ccms_smiles = str(row['Smiles_ccms']).lower()
        lib_smiles = str(row['Smiles_library']).lower()
        noi_smiles = str(row['Smiles_noisy']).lower()
        pre_smiles = str(row['Smiles_predictions']).lower()

        ccms_inchi = str(row['INCHI_ccms']).replace("InChI=","").lower().replace('"','')
        lib_inchi = str(row['INCHI_library']).replace("InChI=","").lower().replace('"','')
        noi_inchi = str(row['INCHI_noisy']).replace("InChI=","").lower().replace('"','')
        pre_inchi = str(row['INCHI_predictions']).replace("InChI=","").lower().replace('"','')

        ccms_name = str(row['Compound_Name_ccms']).lower().replace('"','')
        lib_name = str(row['Compound_Name_library']).lower().replace('"','')
        noi_name = str(row['Compound_Name_noisy']).lower().replace('"','')
        pre_name = str(row['Compound_Name_predictions']).lower().replace('"','')



        if ccms_id == 'nan':
            match_pre.append('no')
            match_noisy.append('no')
            match_lib.append('no')
            continue

        #check on all possiblities for lib spectra
        if str(lib_id) == 'nan':
            match_lib.append('no')
        elif ccms_id == lib_id:
            match_lib.append('yes')
        elif ccms_inchi == lib_inchi:
            match_lib.append('yes')
        elif ccms_smiles == lib_smiles:
            match_lib.append('yes')
        elif ccms_name == lib_name:
            match_lib.append('yes')
        else:
            match_lib.append('unknown')
        
        #pre
        if str(pre_id) == 'nan':
            match_pre.append('no')
        elif ccms_id == pre_id:
            match_pre.append('yes')
        elif ccms_smiles == pre_smiles:
            match_pre.append('yes')
        elif ccms_inchi == pre_inchi:
            match_pre.append('yes')
        elif ccms_name == pre_name:
            match_pre.append('yes')
        else:
            match_pre.append('unknown')
        
        #noisy
        if str(noi_id) == 'nan':
            match_noisy.append('no')    
        elif ccms_id == noi_id:
            match_noisy.append('yes')
        elif ccms_smiles == noi_smiles:
            match_noisy.append('yes')
        elif ccms_inchi == noi_inchi:
            match_noisy.append('yes')
        elif ccms_name == noi_name:
            match_noisy.append('yes')
        else:
            match_noisy.append('unknown')
        



    merge_3['match_predictions'] = match_pre
    merge_3['match_noisy'] = match_noisy
    merge_3['match_library'] = match_lib    
    col_list = ['SharedPeaks_noisy', 'SharedPeaks_predictions', 'SharedPeaks_library', 'Smiles_library',\
    'Smiles_predictions', 'Smiles_ccms', 'Smiles_noisy', 'INCHI_noisy', 'INCHI_ccms', 'INCHI_library', 'INCHI_predictions']
    merge_3.drop(col_list, inplace=True, axis=1)
    print(merge_3)
    merge_3.to_csv("all_merge_GNPS_results.csv")



def currated_sheet_analysis():
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    currated_df = pd.read_csv('all_merge_GNPS_results_currate.csv')
    print(currated_df) 
    added_peaks  = pd.read_csv('./added_peaks_1_10000.csv')
    added_peaks_list = added_peaks['added_peaks'].tolist()

    original_peaks = pd.read_csv('./peak_count_1_10000.csv')
    original_peaks_list = original_peaks['peak_count'].tolist()
    
    score_df = currated_df
    #print(added_peaks_list)
    print(len(added_peaks_list))
    print(len(original_peaks_list))
    score_df['peakos'] = added_peaks_list
    score_df['og_peakos'] = original_peaks_list
    counts = [0.0] * 20
    print(max(added_peaks_list))  
    for item in added_peaks_list:
        counts[item] += 1    

  
    x_val = range(0,20) 
    success_exp = [0.0] * 20
    failed_exp = [0.0] * 20
    lib_exp = [0.0] * 20

    failed_index =[]
    
    
    compound_name = {}
    x_val2 = range(0,94)
    cos_improvement = [0.0] * 20
    cos_plot_norm = [0.0] * 20
    for index, row in score_df.iterrows():
        name = row['SpectrumID_ccms']
        if name not in compound_name:
            compound_name[name] = {}
            compound_name[name]['failed'] = 0
            compound_name[name]['succeeded'] = 0

        lib_m = row['match_library']
        pre_m = row['match_predictions']
        noi_m = row['match_noisy']
        
        
        lib_score = row['MQScore_library']
        pre_score = row['MQScore_predictions']
        noi_score = row['MQScore_noisy']
        
        try:
            peaks = int(row["peakos"])
            peaks2 = int(row['og_peakos'])
        except:
            continue

        if lib_m == 'yes':
            lib_exp[peaks] += 1
        
        
        #criteria 1 defines when a new match is made    
        
        if lib_m == 'no' and pre_m =='yes':
            success_exp[peaks]+=1

        if lib_m == 'yes' and pre_m == 'yes':
            success_exp[peaks] += 1
            compound_name[name]['succeeded'] += 1
        
        if lib_m == 'yes' and pre_m == 'no':
            failed_exp[peaks] += 1
            failed_index.append(index)
            compound_name[name]['failed'] += 1

        """      
        if lib_m == 'no' and noi_m == 'no' and pre_m == 'yes': 
            success_exp[peaks] += 1
            compound_name[name]['succeeded'] += 1
        """
    print(success_exp)
    print(failed_exp)
    
    import matplotlib.patches as mpatches
    x_val = [str(item) for item in x_val]
    #ax = sns.lineplot(x = x_val, y =success_exp, color ='purple', label = 'Succussful Denoising', alpha=5)
    #sns.lineplot(x = x_val, y =failed_exp, color ='orange', label = 'Failed Denoising', alpha = 5)
    #sns.lineplot(x = x_val, y = lib_exp, color = 'green', label = 'Library Matches Constant', alpha = 5)
    
    bar1 = sns.barplot(x=x_val,  y=success_exp, color='purple')
    bar2 = sns.barplot(x=x_val, y=failed_exp, estimator=sum, ci=None,  color='orange')
    
    top_bar = mpatches.Patch(color='purple', label='Denoising Succeed')
    bottom_bar = mpatches.Patch(color='orange', label='Denoising Failed')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.xlabel("Number of Noise Peaks Added")
    plt.ylabel("Number of Correct Library Matches")
    plt.title("Library Annotation Recovery after Noise Addition")

    #ax.set(xlabel='Number of Noise Peaks Added', ylabel='Number of Correct Library Matches')

    plt.show()


def sirius_output_analysis():
    mgf_target = './lib_specs/denoised_40.13.mgf'

    with open('feature_fragment_list.json', 'r') as jf:
        data = json.load(jf)
     
    line_append = False
    temp_list = []
    percent_peaks_explained = {}
    
    with open(mgf_target, 'r') as mgf:
        for line in  mgf:
            if 'END' in line.strip():
                if line_append is True:
                    explained_peaks = [float(item) for item in data[feat_current]]
               
                    if len(explained_peaks) > 1:
                        recovered_signal = [item for item in explained_peaks if item in temp_list]

                        #appears in test, and it should appear in test 
                        percent_peaks_explained[feat_current] =  [explained_peaks,temp_list]
                    line_append = False
                    temp_list = []
            if line_append is True and line.strip()[0].isdigit():
                temp_list.append(float(line.strip().split(' ')[0]))
                        
            if 'FEATURE_ID' in line.strip():
                feat_current = str(line.strip().split('=')[1]) 
                
                if feat_current in data:
                    line_append = True
           
            
                
    with open('percent_peaks_denoised.json', 'w') as jf:
        json.dump(percent_peaks_explained, jf)
def sirius_explore_percent_explained_peaks():
    import seaborn as sns
    import matplotlib.pyplot as plt
    with open('percent_peaks_denoised.json', 'r') as jf:
        denoised_data = json.load(jf)
    
    with open('percent_peaks_noisy.json', 'r') as jf:
        noisy_data = json.load(jf)

    with open('percent_peaks_lib.json', 'r') as jf:
        lib_data = json.load(jf)

    percent_explained_peaks = []
    percent_recover_peaks = []
    for (key,value) in denoised_data.items():
        sirius = value[0]
        spectra = value[1]
        
        explain = [item for item in spectra if item in sirius]
        recover = [item for item in sirius if item in spectra]
        if len(sirius) >= 0 and len(spectra) > 0:
            percent_explained_peaks.append(len(explain)/len(spectra))
            if len(recover) / len(sirius) > 1:
                print(sirius, spectra)
            if len(recover)/len(sirius) < 0.2:
                print(sirius)
                print(spectra)
                print(int(key)%5000, int(key))
            percent_recover_peaks.append(len(recover)/len(sirius))    
            
            #if len(recover)/len(sirius) < 0.5:
        
    sns.set_style('whitegrid')
    label = ['Percent of Recovered Peaks'] * len(percent_recover_peaks)
    nl = ['Percent of Explainable Peaks'] * len(percent_explained_peaks)
    #percent_recover_peaks.extend(percent_explained_peaks)
    #label.extend(nl)
    ax = sns.histplot(x=percent_recover_peaks, hue=label, stat='density', element='step', fill=False)
    print(len(percent_explained_peaks)/100000)
    ax.set(xlabel="Percent of Signal Recovered", ylabel = 'Probability')
    plt.show()
   
if __name__ == "__main__":
    #sirius_output_analysis()
    #sirius_explore_percent_explained_peaks()
    #main()
    currated_sheet_analysis()







