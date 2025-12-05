import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from HybridCORELS import * #got loaf_from_csv
from utils import age_data_modification, get_condition_freq, Data_dir, read_json


def stat_subgroup (data, subgroups, split, method):
    """
    data is read from json file that have all subgroups 
    split = test/train
    method = Pre/Post
    """
    hyperparams = data.keys()
    mean_subgroup = {s:[] for s in subgroups}
    std_subgroup = {s:[] for s in subgroups}
    for i in hyperparams:
        for s in subgroups:
            mean_subgroup[s].append(np.mean(data[i][method][split][s]))
            std_subgroup[s].append(np.std(data[i][method][split][s]))
    
    return mean_subgroup, std_subgroup


def plot_split(means, stds, hyperparams, subgroups, title, sub_freq,savedir):
    plt.figure(figsize=(8,5))
    for s in subgroups:
        m = np.array(means[s])
        sd = np.array(stds[s])
        plt.plot(hyperparams, m, label=f"{s} with total frequency of {sub_freq[s]:.2f}")
        plt.fill_between(hyperparams, m-sd, m+sd, alpha=0.25)

    plt.xlabel("Min. Transparency Constraint")
    plt.ylabel("Transparency Rate")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{savedir}.png")
    plt.show()





if __name__=='__main__':

    condition_gender = ['Gender=Male', 'neg_Gender=Male']
    condition_race = ['Race=African-American', 'Race=Caucasian', 'Race=Hispanic', 'Race=Other'] # all race subgropus in compas
    condition_age = ['Age=18-25','Age=26-29','Age>=30']
    condition = condition_gender # this is the condition that will be applied in statistical analysis

    #know the freq of each subgroups:
    dataset_name = "compas" # Supported: "compas", "adult", "acs_employ"
    # Load data using built-in method
    X, y, features, prediction = load_from_csv("data/%s_mined.csv" %dataset_name)
    #add extra age columns to the data
    X, features = age_data_modification(X,features)
    condition_freq = get_condition_freq(X, features, condition)


    #read all the data for HybridCORELPre and HybridCORELPost
    Method = 'HybridCORELSPreClassifier'  # Supported: "HybridCORELSPreClassifier", "HybridCORELSPostClassifier"

    #data_dir = Path.cwd().parent/ 'examples/'
    

    path_all = np.sort([i for i in Data_dir.iterdir() if i.suffix == '.json' and Method in i.stem ])

    hyperparams = [str(i) for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

    
    split = 'train'
    data = {}
    for i,j in enumerate(path_all):
        data[round(((i+1)*0.1)+0.2, 2)] = read_json(j)

    #calculate the stat for each hyperparameter
    mean_subgroup, std_subgroup = stat_subgroup(data, condition, split, Method)
    
    savedir= 'gender-train-Pre'
    #plot the chart
    plot_split(mean_subgroup, std_subgroup, hyperparams,
                condition, "Train Transparency Rate for Age Subgroups vs Min. Transparecy ",condition_freq, savedir )