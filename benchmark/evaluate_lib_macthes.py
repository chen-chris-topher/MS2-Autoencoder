import sys
import pandas as pd

reg_table = 'lib_2.tsv'
predict_table = 'predict_2.tsv'
noisy_table = 'noisy_2.tsv'
name_table = 'compound_names_2.csv'

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
    with open('ccms_spectra_3.txt') as hf:
        for line in hf:
            ccms_list.append(line.strip())
    ccms_list = ccms_list[:2000]  
    ccms_list_n = ccms_list * 20
    
    num_noise_peaks = []
    
    for i in range(1, len(ccms_list_n)+1):
        num_noise_peaks.append((i // 2000) +1)
    num_noise_peaks[-1] = 20       
    ccms_df = pd.DataFrame({'#Scan#_ccms': range(1, len(ccms_list*20)+1), 'SpectrumID_ccms':ccms_list_n, "Number of Noise Peaks" : num_noise_peaks})
    merge_0 = ccms_df.merge(name, on = 'SpectrumID_ccms', how = 'left')
    merge_1 = merge_0.merge(pre, left_on = '#Scan#_ccms', right_on = '#Scan#_predictions', how = 'outer')
    merge_2 = merge_1.merge(noi, left_on = '#Scan#_ccms', right_on = '#Scan#_noisy', how = 'outer')
    merge_3 = merge_2.merge(reg, left_on = '#Scan#_ccms', right_on = '#Scan#_library', how = 'outer')

    match_pre = []
    match_noisy = []
    match_lib = []

    for index, row in merge_3.iterrows():
        ccms_id = row['SpectrumID_ccms']
        lib_id = row['SpectrumID_library']
        pre_id = row['SpectrumID_predictions']
        noi_id = row['SpectrumID_noisy']
       
        ccms_smiles = row['Smiles_ccms'] 
        lib_smiles = row['Smiles_library']
        noi_smiles = row['Smiles_noisy']
        pre_smiles = row['Smiles_predictions']

        ccms_inchi = str(row['INCHI_ccms']).replace("InChI=","")
        lib_inchi = str(row['INCHI_library']).replace("InChI=","")
        noi_inchi = str(row['INCHI_noisy']).replace("InChI=","")
        pre_inchi = str(row['INCHI_predictions']).replace("InChI=","")

        if (ccms_id == lib_id and str(ccms_id) != 'nan') or (ccms_smiles == lib_smiles and str(ccms_smiles) != 'nan') or (ccms_inchi == lib_inchi and str(ccms_inchi) != 'nan'):
            match_lib.append('yes')
        else:
            match_lib.append('unknown')
        if (ccms_id  == pre_id and str(ccms_id) != 'nan') or (ccms_smiles == pre_smiles and str(ccms_smiles) != 'nan') or (ccms_inchi == pre_inchi and str(ccms_inchi) != 'nan'):
            match_pre.append('yes')
        else:
            match_pre.append('unknown')
        if (ccms_id == noi_id and str(ccms_id) != 'nan') or (ccms_smiles == noi_smiles and str(ccms_smiles) != 'nan') or (ccms_inchi == noi_inchi and str(ccms_inchi) != 'nan'):
            match_noisy.append('yes')
        else:
            match_noisy.append('unknown')
            
    merge_3['match_predictions'] = match_pre
    merge_3['match_noisy'] = match_noisy
    merge_3['match_library'] = match_lib    

    merge_3.to_csv("all_merge_GNPS_results_2.csv")

    #umber_correct_predictions(merge_3)

def number_correct_predictions(merge_3):
    import seaborn as sns
    import matplotlib.pyplot as plt
    x_val = range(1,21) 
    predicted = [0.0] * 20
    lib = [0.0] * 20
    noisy = [0.0] * 20
    
    peaks_added = []
    
    print(len(lib))
    for index, row in merge_3.iterrows():
        
        peaks = row["Number of Noise Peaks"] - 1
            
        lib_val = row['SpectrumID_library']
        pre_val = row['SpectrumID_predictions']
        noi_val = row['SpectrumID_noisy']
        ccms_val = row['SpectrumID_ccms']

        smiles_lib = row['Smiles_library']
        inchi_lib = row['INCHI_library']

        smiles_pre = row['Smiles_predictions']
        inchi_pre = row['INCHI_predictions']

        smiles_noi = row['Smiles_noisy']
        inchi_noi = row['INCHI_noisy']
        
        if ccms_val == lib_val:
            lib[peaks] += 1
        
        if ccms_val == pre_val:
            predicted[peaks] += 1

        if ccms_val == noi_val:
            noisy[peaks] += 1

    
    print(lib)
    print(predicted)
    print(noisy)

    #sns.lineplot(x = x_val, y = lib, color = 'purple')
    sns.lineplot(x = x_val, y = predicted, color = 'orange')
    sns.lineplot(x = x_val, y = noisy, color = 'green')
    plt.show()


def currated_sheet_analysis():
    import seaborn as sns
    import matplotlib.pyplot as plt
    currated_df = pd.read_csv('all_merge_GNPS_results_2.csv')
    print(currated_df)
    #non_currated_df = pd.read_csv('all_merge_GNPS_results.csv', usecols = ['MQScore_library', '#Scan#_ccms', 'MQScore_predictions', 'MQScore_noisy', \
    #'SharedPeaks_library','SharedPeaks_noisy', 'SharedPeaks_predictions'])
    #score_df = currated_df.merge(non_currated_df, on ='#Scan#_ccms', how='left')
    #score_df.to_csv('added_scores.csv')
    
    score_df = currated_df
    
    added_peaks = pd.read_csv('added_peaks_2.csv')
    added_peaks_list = added_peaks['added_peaks'].tolist()
 
    original_peaks = pd.read_csv('peak_count_2.csv')
    original_peaks_list = original_peaks['peak_count'].tolist()
    print(max(added_peaks_list))
    counts = [0.0] * 33
    
    for item in added_peaks_list:
        counts[item] += 1    

   
    score_df['peakos'] = added_peaks_list
    score_df['og_peakos'] = original_peaks_list
    score_df.to_csv("score_df.csv") 
    x_val = range(0,32) 
    success_exp = [0.0] * 32
    failed_exp = [0.0] * 32
    lib_exp = [0.0] * 32

    failed_index =[]
    
    
    compound_name = {}
    x_val2 = range(0,94)
    cos_improvement = [0.0] * 32
    cos_plot_norm = [0.0] * 32
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
        if lib_m == 'yes' and noi_m == 'unknown' and pre_m == 'yes':
            success_exp[peaks] += 1
            compound_name[name]['succeeded'] += 1
        
        if lib_m == 'yes' and noi_m == 'unknown' and pre_m == 'unknown':
            failed_exp[peaks] += 1
            failed_index.append(index)
            compound_name[name]['failed'] += 1

        if lib_m == 'yes' and noi_m == 'yes' and pre_m == 'uknown':
            if noi_score < lib_score:
                failed_exp[peaks] += 1
                failed_index.append(index)
                compound_name[name]['failed'] += 1
               
        if lib_m == 'unknown' and noi_m == 'unknown' and pre_m == 'yes': 
            success_exp[peaks] += 1
            compound_name[name]['succeeded'] += 1
        if lib_m == 'unknown' and noi_m == 'unknown' and pre_m == 'yes':
            success_exp[peaks] += 1
            compound_name[name]['succeeded'] += 1
       
    
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


if __name__ == "__main__":
    #main()
    currated_sheet_analysis()







