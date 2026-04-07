
# March 26,2024, Version 1.0


import numpy as np
from HybridCORELS import *
from pathlib import Path
import pandas as pd
from exp_utils import *
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL
import pickle


# maybe I just start with finding all the models within Rashomon set to see how the number of bootstrap impacts the 
#number of models in the Rashomon set.

# ===============================
# Utilities
# ===============================

def init_trans_total(conditions):
    return {
        split: {
            cond: {
                "ICF": [],
                "FN": {"T": [], "B": []},
                "FP": {"T": [], "B": []},
                "TN": {"T": [], "B": []},
                "TP": {"T": [], "B": []},
                "Pos_Ratio" : {"T": [], "B": []},
            }
            for cond in conditions
        }
        for split in ["train", "test"]
    }




def parse_confusion_matrix(CM):
    """
    CM shape:
    CM[source][true_label, pred_label]
    source: interpretable, Blackbox
    """

    return {
        'T': {  # interpretable
            'TN': int(CM['Interpretable'][0, 0]),
            'FP': int(CM['Interpretable'][0, 1]),
            'FN': int(CM['Interpretable'][1, 0]),
            'TP': int(CM['Interpretable'][1, 1]),
        },
        'B': {  # black-box
            'TN': int(CM['Blackbox'][0, 0]),
            'FP': int(CM['Blackbox'][0, 1]),
            'FN': int(CM['Blackbox'][1, 0]),
            'TP': int(CM['Blackbox'][1, 1]),
        }
    }


def evaluate_group(
    X, y, preds, preds_type, condition, features):
    fairness = FairnessMeasure(X, features, [condition])
    label_T = y[fairness.cond_indices & (preds_type==1)]
    label_B = y[fairness.cond_indices & (preds_type==0)]
    pos_ratio_T = np.sum(label_T)/label_T.shape[0] if label_T.shape[0] > 0 else 0
    pos_ratio_B = np.sum(label_B)/label_B.shape[0] if label_B.shape[0] > 0 else 0
    pos_ratio_all = np.sum(y[fairness.cond_indices])/(y[fairness.cond_indices]).shape[0]
    pos_ratio = {'T':pos_ratio_T,"B":pos_ratio_B}

    icf = fairness.compute_fairness(preds_type, complement=False)['percentage_interpretable']

    CM = fairness.confusion_matrix(
        preds, y, preds_type,
        fairness.cond_indices,
        detailed=True
    )

    return icf, parse_confusion_matrix(CM) , pos_ratio


def update_trans_total(trans_total, split, cond,icf, cm,pos_ratio):

    trans_total[split][cond]['ICF'].append(icf)
    for src in ['T', 'B']:
        trans_total[split][cond]['Pos_Ratio'][src].append(pos_ratio[src])
        for k in ['TP', 'FP', 'TN', 'FN']:
            trans_total[split][cond][k][src].append(cm[src][k])

def evaluate_one_output(X, y,preds, preds_types,split,conditions,features,trans_total):

    acc = float(np.mean(preds == y))
    coverage = float(preds_types.mean())

    for cond in conditions:
        icf, cm, pos_ratio = evaluate_group(
            X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            y,
            preds,
            preds_types,
            cond,
            features
        )
        update_trans_total(trans_total, split, cond, icf, cm, pos_ratio)

    return acc, coverage


seen_predictions = set()
unique_models_preds = []

def store_if_new(y_pred):
    key = y_pred.tobytes()   
    
    if key not in seen_predictions:
        seen_predictions.add(key)
        unique_models_preds.append(y_pred)
        return True   # new model
    return False      # duplicate





##############################



###############################


# in this scripts , I want to load all pickle files , and then filter them based on epsilon and uniqueness

Dataset_name = "compas"
method = "HybridCORELSPostClassifier"
seeed_split = 0
min_transparency = 0.8
epsilon = 0.03

all_models= []
file_path = Path.cwd()/"bootstrap_results"
for f in file_path.iterdir():
    if Dataset_name in f.name and method in f.name and f"seed{seeed_split}" in f.name and f"param{min_transparency}" in f.name:
        with open(f, "rb") as f:
            one_round = pickle.load(f)
            all_models.extend(one_round)



print(f"Number of all bootstraps for {Dataset_name} and {method} and split seed {seeed_split} and transparency {min_transparency}: {len(all_models)}")
whole_data_train_acc = all_models[0]['acc_train']

max_acc = max([i['acc_train'] for i in all_models])


if max_acc - whole_data_train_acc < 0.00001:
    print("Max accuracy is achieved over whole data.")
else:
    print("Max accuracy is significantly better than whole data accuracy.")

unique_preds = {}

for i in all_models:
    model_key = (i['rules'], i['preds_types_train'].tobytes(), i['preds_train'].tobytes())
    if i['acc_train'] > max_acc - epsilon:
        if model_key not in unique_preds:
            unique_preds[model_key] = i.copy()


print(f"Number of unique models within epsilon {epsilon} : {len(unique_preds)}")




