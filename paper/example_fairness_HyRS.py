import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from exp_utils import age_data_modification, save_json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from exp_utils import get_data_norulemining, to_df, FairnessMeasure, Dataset
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL


PARAM_GRID = {
    "Method": [
        "HybridCORELSPreClassifier",
        "HybridCORELSPostClassifier",
        "HyRS",
        "CRL",
    ],
    "Dataset": ["compas", "adult", "acs_employ"],
    "Min_transparency": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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

    icf = fairness.compute_fairness(preds_type, complement=False)['percentage_interpretable']

    CM = fairness.confusion_matrix(
        preds, y, preds_type,
        fairness.cond_indices,
        detailed=True
    )

    return icf, parse_confusion_matrix(CM)




dataset_name = "compas" # Supported: "compas", "adult", "acs_employ"
train_proportion = 0.8
random_state_param = 42

# Load data

my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
print("Before preprocessing",len(my_data.features))
my_data.pre_process()

print("After preprocessing",len(my_data.features))


conditions = my_data.demographicGroup()

condition = conditions['All']
print(condition)


n_seeds = 10

data = {}

for Method in [ "HyRS","CRL"]:
    trans_total = {'train': {i:{'ICF':[],'FN':{'T':[],'B':[]}, 'FP':{'T':[],'B':[]},\
                             'TN':{'T':[],'B':[]}, 'TP':{'T':[],'B':[]}} for i in condition},\
            'test': {i:{'ICF':[],'FN':{'T':[],'B':[]}, 'FP':{'T':[],'B':[]},\
             'TN':{'T':[],'B':[]}, 'TP':{'T':[],'B':[]}}  for i in condition}}
    for i in range(n_seeds):
        np.random.seed(i)
        X, y, features, prediction = my_data.get_data_norulemining(
            {"train" : train_proportion, "test" : 1-train_proportion}, random_state_param = i)

        #convert numpy arrays to dataframes for HyRS and CRL
        if Method in ["HyRS", "CRL"]:
            df_X = my_data.to_df_from_dict(X)

        # Fit a black-box
        bbox = RandomForestClassifier(random_state=i, min_samples_leaf=10, max_depth=10)
        bbox.fit(df_X["train"], y["train"])
        # Test performance
        print("BB Accuracy : ", np.mean(bbox.predict(df_X["test"]) == y["test"]), "\n")
        # Set parameters

        if Method == "CRL":
            hparams = {
                "max_card" : 2,
                "alpha" : 0.01
            }      
        elif   Method == "HyRS":
            hparams = {
                "alpha" : 0.001,
                "beta" : 0.02
            }

        # Define a hybrid model
        if Method == "HyRS":
            hyb_model = HybridRuleSetClassifier(bbox, **hparams)
        elif Method == "CRL":
            hyb_model = CRL(bbox, **hparams)

        # Train the hybrid model
        if Method == "HyRS":
            hyb_model.fit(df_X["train"], y["train"], 5000, T0=0.01, premined_rules=True, 
                                                    random_state=i, time_limit=30)
        elif Method == "CRL":
            hyb_model.fit(df_X["train"], y["train"], n_iteration=50000, random_state=i+1, 
                                                            premined_rules=True, time_limit=10)
        #Evaluate train performance 
        preds_train, preds_types_train = hyb_model.predict_with_type(df_X["train"])
        preds_types_counts_train = np.unique(preds_types_train, return_counts=True)
        index_one_train = np.where(preds_types_counts_train[0] == 1)
        cover_rate_train = preds_types_counts_train[1][index_one_train][0]/np.sum(preds_types_counts_train[1])
        print("=> Training accuracy = ", np.mean(preds_train == y["train"]))
        print("=> Training transparency = ", cover_rate_train)

        # Evaluate test performances
        print(hyb_model.get_description(df_X["test"], y["test"]))
        preds_test, preds_types_test = hyb_model.predict_with_type(df_X["test"])
        preds_types_counts_test = np.unique(preds_types_test, return_counts=True)
        index_one_test = np.where(preds_types_counts_test[0] == 1)
        cover_rate_test = preds_types_counts_test[1][index_one_test][0]/np.sum(preds_types_counts_test[1])
        # print("=> Test accuracy = ", np.mean(preds_test == y_test))
        # print("=> Test transparency = ", cover_rate_test)

        for cond in condition:

            # ---- TRAIN ----
            icf, cm = evaluate_group(
                X["train"], y["train"],
                preds_train, preds_types_train,
                cond, features
            )

            trans_total['train'][cond]['ICF'].append(icf)

            for src in ['T', 'B']:
                for k in ['TP', 'FP', 'TN', 'FN']:
                    trans_total['train'][cond][k][src].append(cm[src][k])

            # ---- TEST ----
            icf, cm = evaluate_group(
                X["test"], y["test"],
                preds_test, preds_types_test,
                cond, features
            )

            trans_total['test'][cond]['ICF'].append(icf)
            
            for src in ['T', 'B']:
                for k in ['TP', 'FP', 'TN', 'FN']:
                    trans_total['test'][cond][k][src].append(cm[src][k])

    data[Method] = trans_total

print(data)


