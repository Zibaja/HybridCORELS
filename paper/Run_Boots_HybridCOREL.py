### In these lines , I would like to run experimnets for one dataset and one method , with bootstrap sampling ###

# April 6, 

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
            **h["corels_params"]
        ),
        "fit": lambda model, X, y, h: model.fit(X, y, features=h["features"],
                                                                prediction_name=h['prediction_name'], time_limit=h["time_limit"],
                                                                memory_limit=h["memory_limit"]),
        "hparams": {
            "alpha": 2,
            "lambdaValue" : 0.001,
            "beta": lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2]),
            "memory_limit": 4000,
            "min_coverage": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95]
        },
    },

    "HybridCORELSPostClassifier": {
        "build": lambda bbox, h: HybridCORELSPostClassifier(
            black_box_classifier=bbox,
            beta=h["beta"],
            c = h["lambdaValue"],
            min_coverage=h["min_coverage"],
            bb_pretrained=False,
            **h["corels_params"]
        ),
        "fit": lambda model, X, y, h: model.fit(X, y, features=h["features"],
                                                                prediction_name=h['prediction_name'], time_limit=h["time_limit"],
                                                                memory_limit=h["memory_limit"]),
        "hparams": {
            "beta": lambda X,lambdaValue : min([ (1 / X.shape[0]) / 2, lambdaValue / 2]),
            "lambdaValue" : 0.001,
            "memory_limit": 4000,
            "min_coverage": [0.1,0.2,0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,0.95]
        },
    },

    "HyRS": {
        "build": lambda bbox, h: HybridRuleSetClassifier(
            bbox,
            alpha=h["alpha"],
            beta=h["beta"]
        ),
        "fit": lambda model, X, y, h: model.fit(
            X, y,
            h["n_iteration"],
            T0=h["T0"],
            premined_rules=True,
            random_state=h["seed"],
            time_limit=h["time_limit"]
        ),
        "hparams": {
            "alpha": 0.001,
            "beta": [0.001, 0.00215443, 0.00464159, 0.01, 0.02154435,
                        0.04641589, 0.1, 0.21544347, 0.46415888, 1.0], #changed from 0.02 to 0.1 
            "n_iteration": 50000,
            "T0": 0.01,
        },
    },

    "CRL": {
        "build": lambda bbox, h: CRL(
            bbox,
            max_card=h["max_card"],
            alpha=h["alpha"]
        ),
        "fit": lambda model, X, y, h: model.fit(
            X, y,
            n_iteration=h["n_iteration"],
            random_state=h["seed"],
            premined_rules=True,
            time_limit=h["time_limit"]
        ),
        "hparams": {
            "max_card": 2,
            "alpha": [0.001, 0.0016681, 0.00278256, 0.00464159, 0.00774264,
                        0.0129155,0.02154435, 0.03593814, 0.05994843, 0.1], ##I think 0.01 is better 
            "n_iteration": 50000,
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
    "HybridCORELSPreClassifier": "min_coverage",
    "HybridCORELSPostClassifier": "min_coverage",
    "CRL": "alpha",
    "HyRS": "beta",
}

TRADEOFF_VALUES = {"HybridCORELSPreClassifier": ESTIMATORS["HybridCORELSPreClassifier"]["hparams"]["min_coverage"],
                   "HybridCORELSPostClassifier": ESTIMATORS["HybridCORELSPostClassifier"]["hparams"]["min_coverage"],
                   "CRL": ESTIMATORS["CRL"]["hparams"]["alpha"],
                   "HyRS": ESTIMATORS["HyRS"]["hparams"]["beta"]}

# ===============================
# Experiment grid (FLATTENED)
# ===============================

n_seeds = 5
DATASETS = ["compas", "adult", "acs_employ"]
SEEDS = list(range(n_seeds))
round = list(range(10))

EXPERIMENTS = []

for dataset in DATASETS:
    for model in ["HybridCORELSPreClassifier","HybridCORELSPostClassifier"]:
        for tardeoff_value in TRADEOFF_VALUES[model]:
            for seed in SEEDS:
                for r in round:
                    EXPERIMENTS.append({
                        "dataset": dataset,
                        "model": model,
                        TRADEOFF_PARAM[model]: tardeoff_value,
                        "seed": seed,
                        "round" : r
                    })

# for i,j in enumerate(EXPERIMENTS):
#     if j['dataset']=='compas' and j['model']=='HybridCORELSPostClassifier' and j['seed']==0:
#         if j['round']<5:
#             print(i,j)
# print(len(EXPERIMENTS))





##############################



###############################



def run_one_bootsrap_batch(cfg,round_number, n_batch=100):
    dataset_name = cfg["dataset"]
    model_key = cfg["model"]
    tradeoff_param = TRADEOFF_PARAM[model_key]
    tradeoff_value = cfg[tradeoff_param]
    seed = cfg["seed"] #test/train split seed
    time_limit=300 #5 min
    np.random.seed(seed)
    
    #split information
    train_proportion=0.8

    # Load data
    my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
    my_data.pre_process()


    X, y, features, prediction = my_data.get_data_norulemining(
                            {"train" : train_proportion, "test" : 1-train_proportion}, random_state_param = seed)




    spec = ESTIMATORS[model_key]
    h = spec["hparams"].copy()

    h.update({
        "features": features,
        "time_limit": time_limit,
        "corels_params": CORELS_PARAMS,
        "prediction_name": prediction,
    })

    h[TRADEOFF_PARAM[model_key]] = tradeoff_value #run one hyperparametre 


    if callable(h.get("beta", None)):
        h["beta"] = h["beta"](X["train"], h["lambdaValue"]) #it should be changed in each iteration of the bootstrap sampling, but for now I will keep it fixed to save time


    #Build and fit the model over whole training data
    # Fit a black-box
    bbox = RandomForestClassifier(random_state=seed, min_samples_leaf=10, max_depth=10)
    model = spec["build"](bbox, h)
    spec["fit"](model, X["train"], y["train"], h)
    rules = [i['antecedents'][0]-1 for i in model.interpretable_part.rl().rules][:-1] #last one is the default rule, so I remove it


    if model_key in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
        status = model.get_status()
    else:
        status = None

        
 

    #predict for train and test
    preds_train, preds_types_train= model.predict_with_type(X["train"])
    preds_test, preds_types_test = model.predict_with_type(X["test"])

    acc_train = float(np.mean(preds_train == y["train"]))
    acc_test = float(np.mean(preds_test == y["test"])) 
    coverage_rate_train = float(preds_types_train.mean())
    coverage_rate_test = float(preds_types_test.mean()) 


    #store each model
    all_models = []

    all_models.append({
    "test-train-split-seed": seed,
    "round_number": round_number,
    "bootstrap_id": 0,
    "rules": tuple(rules),
    "preds_train": preds_train.astype(np.uint8),
    "preds_types_train": preds_types_train.astype(np.uint8),
    "preds_test": preds_test.astype(np.uint8),
    "preds_types_test": preds_types_test.astype(np.uint8),
    "acc_train": acc_train,
    "acc_test": acc_test,
    "coverage_rate_train": coverage_rate_train,
    "coverage_rate_test": coverage_rate_test,  
    "status": status,
    })


    B = n_batch  # number of bootstrap models
    n = len(X['train'])
    

    for b in [(i)+(round_number*B) for i in range(1,B+1)]:
        np.random.seed(b)  # different seed each time
        
        # sample indices with replacement
        indices = np.random.choice(n, size=n, replace=True)
        
        X_boot = X['train'][indices]
        y_boot = y['train'][indices]
        
        #build and fit the model on  the bootstrap sample

        bbox = RandomForestClassifier(random_state=b, min_samples_leaf=10, max_depth=10)
        model = spec["build"](bbox, h)
        #fit the model on  the bootstrap sample
        spec["fit"](model, X_boot, y_boot, h)
        rules = [i['antecedents'][0]-1 for i in model.interpretable_part.rl().rules][:-1] #last one is the default rule, so I remove it


        if model_key in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
            status = model.get_status()
        else:
            status = None

        

        #check predictions on the original training data and test data
        preds_train, preds_types_train= model.predict_with_type(X['train'])
        preds_test, preds_types_test = model.predict_with_type(X["test"])

        acc_train = float(np.mean(preds_train == y["train"]))
        acc_test = float(np.mean(preds_test == y["test"])) 
        coverage_rate_train = float(preds_types_train.mean())
        coverage_rate_test = float(preds_types_test.mean()) 


        all_models.append({
        "test-train-split-seed": seed,
        "round_number": round_number,
        "bootstrap_id": b,
        "rules": tuple(rules),
        "preds_train": preds_train.astype(np.uint8),
        "preds_types_train": preds_types_train.astype(np.uint8),
        "preds_test": preds_test.astype(np.uint8),
        "preds_types_test": preds_types_test.astype(np.uint8),
        "acc_train": acc_train,
        "acc_test": acc_test,
        "coverage_rate_train": coverage_rate_train,
        "coverage_rate_test": coverage_rate_test,  
        "status": status,
        })




    return all_models



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run bootstrap experiments for one dataset, model, and seed.')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--round_min', type=int, default=None)
    parser.add_argument('--round_max', type=int, default=None)
    parser.add_argument('--local_id', type=int, required=True)

    args = parser.parse_args()

    filtered_experiments = []

    for cfg in EXPERIMENTS:
        if args.dataset is not None and cfg["dataset"] != args.dataset:
            continue
        if args.model is not None and cfg["model"] != args.model:
            continue
        if args.seed is not None and cfg["seed"] != args.seed:
            continue
        if args.round_min is not None and cfg["round"] < args.round_min:
            continue
        if args.round_max is not None and cfg["round"] > args.round_max:
            continue

        
        filtered_experiments.append(cfg)
        # print(cfg, filtered_experiments.index(cfg))
     

    #print(f"Total filtered jobs: {len(filtered_experiments)}")

    cfg = filtered_experiments[args.local_id]

    #print(f"Running configuration: {cfg}")
    results = run_one_bootsrap_batch(cfg, round_number=cfg['round'], n_batch=100)
    
    # Save results 
    output_dir = Path.cwd() / 'bootstrap_results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / (
    f"{cfg['dataset']}_"
    f"{cfg['model']}_"
    f"seed{cfg['seed']}_"
    f"round{cfg['round']}_"
    f"param{cfg[TRADEOFF_PARAM[cfg['model']]]}.pkl")

    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    






