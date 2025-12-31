import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from HybridCORELS import * #got loaf_from_csv
from utils import *






def stat_subgroup (data, subgroups, split, method):
    """This function calculates the mean, std and se of the ICF of all seeds for all conditions
      and all Hyperparameters (min_coverage)
    for example is subgroups = ['Male','Female'] , the output is 
    Args:
        data (_type_): _description_
        subgroups (_type_): _description_
        split (_type_): _description_
        method (_type_): _description_

    Returns:
        _type_: _description_
    """
    hyperparams = data.keys()
    mean_subgroup = {s:[] for s in subgroups}
    std_subgroup = {s:[] for s in subgroups}
    for i in hyperparams:
            for s in subgroups:
                mean_subgroup[s].append(np.mean(data[i][method][split][s]['ICF']))
                std_subgroup[s].append(np.std(data[i][method][split][s]['ICF']))
    se_subgroup = {k:np.array(std_subgroup[k])/np.sqrt(len(data[i][method][split][s]['ICF'])) for k in subgroups}
        
    return mean_subgroup, std_subgroup, se_subgroup


def fairness_subgroup (data,subgroups, split, Method, model_part = None, signed = True ):
    """This function calculates mean, std and se of TPR gap of all seeds for a demographic group
    for a hyperparameter 
    Args:
        data (dict): the data which is read from json file
        subgroups (list): list of conditions of a demographic group like ['Male','Female']
        split (str): test/train
        Method (str): HybridPre/ HybridPost
        model_part (str, optional): Does the disparity come from overall model
        or only from BB or Interpretable part . Defaults to None.

    Returns:
        dict: mean, std and se of all fairness gaps in seeds
    """
    hyperparams = data.keys()
    fairness = {'mean': [], 'std': [], 'se':[]}
    for i in hyperparams:
        TPR_info = get_TPR(data[i][Method][split])
        if signed : 
            fairness_gap = compute_signed_global_fairness(TPR_info, subgroups,model_part= model_part) #signed 
        else:
            fairness_gap = compute_global_fairness(TPR_info, subgroups,model_part= model_part) #signed 
        mean, std, se = statistics(fairness_gap)
        fairness['mean'].append(mean)
        fairness['std'].append(std)
        fairness['se'].append(se)
    return fairness




def plot_split(means, stds,se, hyperparams, subgroups, title, sub_info,savedir):
    plt.figure(figsize=(8,5))
    for s in subgroups:
        m = np.array(means[s])
        sd = np.array(stds[s])
        CI = np.array(se[s]) *1.96
        plt.plot(hyperparams, m, label=f"{s} frequency={sub_info['freq'][s]:.2f}, correlation={sub_info['corr'][s]:.1f}")
        plt.fill_between(hyperparams, m-CI, m+CI, alpha=0.25)

    plt.xlabel("Min. Transparency Constraint")
    plt.ylabel("Transparency Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{savedir}.png")
    #plt.show()



def plot_fairness( hyperparams,condition, disparity,savedir,overall, signed ):
    plt.figure(figsize=(8,5))
    #print(f"This is condition{condition}")
    if overall:
        fairness = fairness_subgroup (data,condition, split, Method, model_part = None, signed=signed )
        m = fairness['mean']
        CI = np.array(fairness['se'])*1.96
        plt.plot(hyperparams, m)
        plt.fill_between(hyperparams, m-CI, m+CI, alpha=0.25)

    else:
        for i in ['T','B']:
            model_part = i
            fairness = fairness_subgroup (data,condition, split, Method, model_part, signed=signed )
            m = fairness['mean']
            CI = np.array(fairness['se'])*1.96
            plt.plot(hyperparams, m, label=f"TPR gap coming from {'interpretable part' if model_part=='T' else 'blackbox'}")
            plt.fill_between(hyperparams, m-CI, m+CI, alpha=0.25)
            plt.legend()

    plt.xlabel("Min. Transparency Constraint")
    plt.ylabel(f"TPR fairness gap for {disparity}")
    plt.title(f"TPR Disparity for {disparity}" if not signed else f'Signed TPR difference for {disparity}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{savedir}.png")
    #plt.show()



#TODO : try to reconstruct the structure of the functions and analysis
if __name__=='__main__':
    
    condition_gender = ['Gender=Male', 'neg_Gender=Male']
    condition_race = ['Race=African-American', 'Race=Caucasian', 'Race=Hispanic', 'Race=Other'] # all race subgropus in compas
    condition_age = ['Age=18-25','Age=26-29','Age>=30']
    condition = condition_age # this is the condition that will be applied in the analysis

    if condition == condition_age:
        disparity = 'Age'
    elif condition == condition_race:
        disparity = 'Race'
    elif condition == condition_gender:
        disparity = 'Gender'

    #know the freq of each subgroups:
    dataset_name = "compas" # Supported: "compas", "adult", "acs_employ"
    # Load data using built-in method
    X, y, features, prediction = load_from_csv("data/%s_mined.csv" %dataset_name)
    #add extra age columns to the data
    X, features = age_data_modification(X,features)

    #analysis of the filtered data
    demographic_data = DemographicData(X,y,features,condition)
    condition_freq = demographic_data.get_condition_freq()
    condition_corr = demographic_data.corr_subgroups()
    sub_info = {'freq': condition_freq, 'corr': condition_corr}


    data = read_json(Data_dir/ 'fairnesswithFNFPNEWALL.json')
    hyperparams = data.keys()
    for Method in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
        for split in ['train', 'test']:
            mean_subgroup, std_subgroup, se_subgroup = stat_subgroup (data, condition, split, Method)
           
            savedir= f'ICF for {disparity}-{split}-{Method}'
            #plot the chart for ICF
            title = f"{'Train' if split=='train' else 'Test'} Transparency Rate for {disparity} Subgroups vs Min. Transparecy "
            plot_split(mean_subgroup, std_subgroup,se_subgroup, hyperparams,\
                        condition,title ,sub_info, savedir )
            
            #plot the chart for TPR gap overall
            savedir= f'Disparity of {disparity}-overall-{split}-{Method}'
            plot_fairness(hyperparams,condition, disparity,savedir,overall = True, signed = False)
            #plot the chart for TPR gap based on each model part
            savedir= f'Disparity of {disparity}-Detailed-{split}-{Method}'
            plot_fairness(hyperparams,condition ,disparity,savedir,overall = False, signed = False)
            for p in paired_subgroups(condition):
                #print(p, split, Method)

                if 'Age' in p[0]:
                    disparity = 'Age'
                elif 'Race' in p[0]:
                    disparity = 'Race'
                elif 'Gender' in p[0]:
                    disparity = 'Gender'
                

                 #plot the chart for signed TPR gap overall 
                savedir= f'Signed Disparity of {disparity}-overall-{split}-{Method}-{p}'
                plot_fairness(hyperparams,p, disparity,savedir,overall = True, signed = True)
                #plot the chart for signed TPR gap based on each model part
                savedir= f'Signed Disparity of {disparity}-Detailed-{split}-{Method}-{p}'
                plot_fairness(hyperparams,p, disparity,savedir,overall = False, signed = True)

                

