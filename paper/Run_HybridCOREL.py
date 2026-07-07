"""
This code is to run only HybridCOREL methods with 
max prefix lenghth to compare the perfromance with HybridDT
date: 2026-07-07


"""

from copyreg import pickle
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
import pickle


# ===============================
# ESTIMATORS dictionary here:
# ===============================

ESTIMATORS = {

    "HybridCORELSPreClassifier": {
        "build": lambda bbox, h: HybridCORELSPreClassifier(
            black_box_classifier=bbox,
            beta=h["beta"],
            c = h["lambdaValue"],
            alpha=h["alpha"],
            min_coverage=h["min_coverage"],
            obj_mode='collab',
            max_length=h["max_lenght"],
            **h["corels_params"]
        ),
        "fit": lambda model, X, y, h: model.fit(X, y, features=h["features"],
                                                                prediction_name=h['prediction_name'], time_limit=h["time_limit"],
                                                                memory_limit=h["memory_limit"]),
        "hparams": {
            "alpha": 2,
            "lambdaValue" : [10**-2, 10**-3, 10**-4],
            "beta": lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2]),
            "memory_limit": 8000,
            "min_coverage": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95],
            "max_lenght": [2**i for i in [1,2,3,4,5]]
        },
    },

    "HybridCORELSPostClassifier": {
        "build": lambda bbox, h: HybridCORELSPostClassifier(
            black_box_classifier=bbox,
            beta=h["beta"],
            c = h["lambdaValue"],
            min_coverage=h["min_coverage"],
            bb_pretrained=False,
            max_length=h["max_lenght"],
            **h["corels_params"]
        ),
        "fit": lambda model, X, y, h: model.fit(X, y, features=h["features"],
                                                                prediction_name=h['prediction_name'], time_limit=h["time_limit"],
                                                                memory_limit=h["memory_limit"]),
        "hparams": {
            "beta": lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2]),
            "lambdaValue" : [10**-2, 10**-3, 10**-4],
            "memory_limit": 8000,
            "min_coverage": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95],
            "max_lenght": [2**i for i in [1,2,3,4,5]],
        },
    },
}

#Shared CORELS parameters
CORELS_PARAMS = {
    "policy": "objective",
    "max_card": 1,
    "n_iter": 10**9,
    'min_support':0.05,
    "verbosity": ["hybrid"],
}

# Trade-off parameter name for each model
TRADEOFF_PARAM = {
    "HybridCORELSPreClassifier": ["min_coverage", "lambdaValue", "max_lenght"],
    "HybridCORELSPostClassifier": ["min_coverage", "lambdaValue", "max_lenght"],
  
}

TRADEOFF_VALUES = {"HybridCORELSPreClassifier":
                   {k:ESTIMATORS["HybridCORELSPreClassifier"]["hparams"][k] for k in TRADEOFF_PARAM["HybridCORELSPreClassifier"] },
                   "HybridCORELSPostClassifier": {k:ESTIMATORS["HybridCORELSPostClassifier"]["hparams"][k] for k in TRADEOFF_PARAM["HybridCORELSPostClassifier"] }
}

# ===============================
# Experiment grid (FLATTENED)
# ===============================

n_seeds = 5
DATASETS = ["compas", "adult", "acs_employ"]
SEEDS = [1,2,3,4,5]


EXPERIMENTS = []

for dataset in DATASETS:
    for model in ['HybridCORELSPostClassifier']:
        for min_cov in TRADEOFF_VALUES['HybridCORELSPostClassifier']['min_coverage']:
            for depth in TRADEOFF_VALUES['HybridCORELSPostClassifier']['max_lenght']:
                for lambda_val in TRADEOFF_VALUES['HybridCORELSPostClassifier']['lambdaValue']:
                    for seed in SEEDS:
                        EXPERIMENTS.append({
                            "dataset_name": dataset,
                            "model": model,
                            "min_coverage": min_cov,
                            "max_lenght": depth,
                            "lambdaValue": lambda_val,
                            "seed": seed,
                        })

# for i,j in enumerate(EXPERIMENTS):
#     if j['dataset']=='compas' and j['model']=='HybridCORELSPostClassifier' and j['seed']==0:
#         if j['round']<5:
#             print(i,j)
# print(len(EXPERIMENTS))





##############################



###############################



def run_one_model(time_limit,dataset_name, model,min_coverage,max_lenght,lambdaValue ,seed):


    np.random.seed(seed)
    
    # #split information
    # train_proportion=0.8

    # Load data
    my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
    
    splits = {"train": 2000, "test":2000, "validation":2000}
    X, y, features, prediction = my_data.split_data_as_dict_withsize(splits, random_state_param = seed)

               

    # Fit a black-box
    random_state = 42+seed
    bbox = RandomForestClassifier(random_state=random_state, min_samples_leaf=10, max_depth=10)
    bbox.fit(X["train"], y["train"])

    spec = ESTIMATORS[model]
    h={
        "features": features,
        "time_limit": time_limit,
        "memory_limit": ESTIMATORS[model]["hparams"]["memory_limit"],
        "min_coverage": min_coverage,
        "max_lenght": max_lenght,
        "lambdaValue": lambdaValue,
        "beta" : ESTIMATORS[model]["hparams"]["beta"](X["train"], lambdaValue) ,
        "corels_params": CORELS_PARAMS,
        "prediction_name": prediction
    }




    #Build and fit the model over whole training data
    hybridmodel = spec["build"](bbox, h)
    spec["fit"](hybridmodel, X["train"], y["train"], h)

    if model in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
        rules = tuple([i['antecedents'][0]-1 for i in hybridmodel.interpretable_part.rl().rules][:-1] )#last one is the default rule, so I remove it
        status = hybridmodel.get_status()
        print("status of CORELS is : ", status)
        




    #predict for train and test
    preds_train, preds_types_train= hybridmodel.predict_with_type(X["train"]) 
    preds_test, preds_types_test = hybridmodel.predict_with_type(X["test"])

    acc_train = float(np.mean(preds_train == y['train']))
    acc_test = float(np.mean(preds_test == y["test"])) 
    coverage_rate_train = float(preds_types_train.mean())
    coverage_rate_test = float(preds_types_test.mean()) 
        #store each model
    results = [{
    "test-train-split-seed": seed,
    "dataset_name": dataset_name,
    "model": model,
    "min_coverage": min_coverage,
    "max_lenght": max_lenght,
    "lambdaValue": lambdaValue,
    "rules": rules,
    "preds_train": preds_train.astype(np.uint8),
    "preds_types_train": preds_types_train.astype(np.uint8),
    "preds_test": preds_test.astype(np.uint8),
    "preds_types_test": preds_types_test.astype(np.uint8),
    "acc_train": acc_train,
    "acc_test": acc_test,
    "coverage_rate_train": coverage_rate_train,
    "coverage_rate_test": coverage_rate_test,  
    "status": status,
    }]
  
    print(results)
    return results



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run bootstrap experiments for one dataset, model, and seed.')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--local_id', type=int, required=True)

    args = parser.parse_args()

    filtered_experiments = []

    for cfg in EXPERIMENTS:
        if args.dataset is not None and cfg["dataset_name"] != args.dataset:
            continue
        if args.model is not None and cfg["model"] != args.model:
            continue
        if args.seed is not None and cfg["seed"] != args.seed:
            continue

        
        filtered_experiments.append(cfg)
        # print(cfg, filtered_experiments.index(cfg))
     
    
    #print(f"Total filtered jobs: {len(filtered_experiments)}")

    cfg = filtered_experiments[args.local_id]
    #print(f"Running configuration: {cfg}")
    results = run_one_model(time_limit=3600, **cfg)
    
    # Save results 
    output_dir = Path.cwd() / 'HybridCORELS_results' 
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / (
    f"{cfg['dataset_name']}_"
    f"{cfg['model']}_"
    f"seed{cfg['seed']}_"
    f"depth{cfg['max_lenght']}_"
    f"min_cov{cfg['min_coverage']}_"
    f"lambdaValue{cfg['lambdaValue']}.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)



    
    






