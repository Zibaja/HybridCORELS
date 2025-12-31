import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from HybridCORELS import * #got loaf_from_csv
from utils import *
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import pandas as pd





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


if __name__=='__main__':

    condition_gender = ['Gender=Male', 'neg_Gender=Male']
    condition_race = ['Race=African-American', 'Race=Caucasian', 'Race=Hispanic', 'Race=Other'] # all race subgropus in compas
    condition_age = ['Age=18-25','Age=26-29','Age>=30']
    condition = condition_race # this is the condition that will be applied in the analysis

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
    
    #reference group is the most frequent subgroup 
    reference = condition[np.argmax(condition_freq.values())]
    data = read_json(Data_dir/ 'fairnesswithFNFPNEWALL.json')
    hyperparams = np.array([float(i) for i in data.keys()])
    for Method in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
        for split in ['train', 'test']:
            #calculate the stat for each hyperparameter
            mean_subgroup, _, _ = stat_subgroup(data, condition, split, Method)

            for k in [[i, reference] for i in condition if i!=reference]:
                #TPR difference between the target and reference group 
                delta_TPR = fairness_subgroup (data,k, split, Method, model_part = None, signed=True )['mean']
                
                pearson_r, pearson_p = pearsonr(mean_subgroup[k[0]], delta_TPR)
                spearman_r, spearman_p = spearmanr(mean_subgroup[k[0]], delta_TPR)

                print(f"Pearson r = {pearson_r:.2f} (p={pearson_p:.3f})")
                print(f"Spearman ρ = {spearman_r:.2f} (p={spearman_p:.3f})")



                # Your data
                ICF = mean_subgroup[k[0]] #the average ICF for the target group
                df = pd.DataFrame({
                    "ICF": ICF,
                    "delta_TPR": delta_TPR,
                    "tau": hyperparams
                })
                savedir = f"{k}-{split}-{Method}"
                plt.figure(figsize=(6.5,5.5))
                # Regression line + 95% CI (no scatter)
                sns.regplot(
                    data=df,
                    x="ICF",
                    y="delta_TPR",
                    scatter=False,
                    ci=95,
                    line_kws={"color": "black", "linewidth": 2}
                )
                #Scatter plot with color = hyperparameter
                sc = plt.scatter(
                    df["ICF"],
                    df["delta_TPR"],
                    c=df["tau"],
                    cmap="viridis",
                    s=130,
                    edgecolor="black"
                )
                #Optional: connect points to show progression
                plt.plot(
                    df["ICF"],
                    df["delta_TPR"],
                    linestyle="--",
                    alpha=0.5,
                    color="gray"
                )
                # Colorbar
                cbar = plt.colorbar(sc)
                cbar.set_label("Transparency Constraint")

                # Reference line
                plt.axhline(0, linestyle="--", color="gray")

                # Labels & title
                plt.xlabel(f"Interpretability Coverage (ICF) - {k[0]}")
                plt.ylabel(f"Signed TPR Difference ({k[0]}-{k[1]})")
                plt.title(f"Relationship between Interpretability Coverage and TPR Disparity-{split}-{Method}")

                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"{savedir}.png")
                plt.show()

            




