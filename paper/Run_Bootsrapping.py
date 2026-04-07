### In these lines , I would like to run experimnets for one dataset and one method , with bootstrap sampling ###

# March 26,2024, Version 1.0


import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from exp_utils import *
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL
import argparse


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

dataset_name = 'compas'
model_key = 'HybridCORELSPreClassifier'
tradeoff_param = TRADEOFF_PARAM[model_key]
tradeoff_value = 0.8
n_seeds = 10
SEEDS = list(range(n_seeds))
seed = SEEDS[3] #just to test the code for the first split, I will change it later to loop over all seeds

time_limit=3600 #to be changed later
#split information
train_proportion=0.8



np.random.seed(seed)

# Load data
my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
my_data.pre_process()

conditions = my_data.demographicGroup()
conditions = conditions['All']
trans_total = init_trans_total(conditions)
X, y, features, prediction = my_data.get_data_norulemining(
                        {"train" : train_proportion, "test" : 1-train_proportion}, random_state_param = seed)

#convert numpy arrays to dataframes for HyRS and CRL
df_X = my_data.to_df_from_dict(X)

# Fit a black-box
bbox = RandomForestClassifier(random_state=seed, min_samples_leaf=10, max_depth=10)
#bbox.fit(X["train"], y["train"])


spec = ESTIMATORS[model_key]
h = spec["hparams"].copy()

h.update({
    "features": features,
    "seed": seed,
    "time_limit": time_limit,
    "corels_params": CORELS_PARAMS,
    "prediction_name": prediction,
})

h[TRADEOFF_PARAM[model_key]] = tradeoff_value #run one hyperparametre 

if callable(h.get("beta", None)):
    h["beta"] = h["beta"](X["train"], h["lambdaValue"])


#Build and fit the model over whole training data
model = spec["build"](bbox, h)
spec["fit"](model, X["train"], y["train"], h)
rules = [i['antecedents'][0]-1 for i in model.interpretable_part.rl().rules][:-1]


if model_key in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
    status = model.get_status()
else:
    status = None

acc = {"train": [], "test": []}
cov = {"train": [], "test": []}

#predict for train and test
preds_train, preds_types_train= model.predict_with_type(X["train"])
preds_test, preds_types_test = model.predict_with_type(X["test"])



acc_train, coverage_rate_train = evaluate_one_output(X["train"], y["train"],preds_train, preds_types_train,"train", conditions, features, trans_total)
acc["train"].append(acc_train) 
cov["train"].append(coverage_rate_train)
opt_acc = acc_train
opt_preds_train = preds_train
opt_preds_types_train = preds_types_train

#for test
acc_test, coverage_rate_test = evaluate_one_output(X["test"], y["test"],preds_test, preds_types_test ,"test", conditions, features, trans_total)
acc["test"].append(acc_test)
cov["test"].append(coverage_rate_test)

unique_preds = {}
rules_key = tuple(rules)
pred_type_key = preds_types_train.tobytes()
pred_key = preds_train.tobytes()
model_key = (rules_key, pred_type_key, pred_key)
key = preds_train.tobytes()

if key not in unique_preds:
    unique_preds[key] = preds_train.copy()

B = 5  # number of bootstrap models
n = len(X['train'])


epsilon = 0.01
for b in range(B):
    np.random.seed(b)  # different seed each time
    
    # sample indices with replacement
    indices = np.random.choice(n, size=n, replace=True)
    
    X_boot = X['train'][indices]
    y_boot = y['train'][indices]
    #fit the model on  the bootstrap sample
    spec["fit"](model, X_boot, y_boot, h)


    

    #check predictions on the original training data and test data
    preds_train, preds_types_train= model.predict_with_type(X['train'])
    preds_test, preds_types_test = model.predict_with_type(X["test"])


    acc_train_model = float(np.mean(preds_train == y['train'])) 
    coverage = float(preds_types_train.mean())
    
    if acc_train_model > opt_acc:
        opt_acc = acc_train_model
        opt_preds_train = preds_train
        opt_preds_types_train = preds_types_train

    key = preds_train.tobytes() #do I need to check routing as well?
    if acc_train_model >= opt_acc - epsilon : 
        if key not in unique_preds:
            unique_preds[key] = preds_train.copy()

            acc_train, coverage_rate_train = evaluate_one_output(X['train'], y['train'],preds_train, preds_types_train,"train", conditions, features, trans_total)
            acc["train"].append(acc_train) 
            cov["train"].append(coverage_rate_train)

            #for test
            acc_test, coverage_rate_test = evaluate_one_output(X["test"], y["test"],preds_test, preds_types_test ,"test", conditions, features, trans_total)
            acc["test"].append(acc_test)
            cov["test"].append(coverage_rate_test)

print(f"optimal acc is {opt_acc}")
print(50*"-")
print('train accuracy is...')
print(acc["train"])
print(50*"-")
print('test accuracy is...')
print(acc["test"])

