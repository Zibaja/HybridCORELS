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
import argparse
import pickle
import time
import subprocess
from black_box_models import BlackBox


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
    #read bb from pretrained models are only considered for Post, should be changed later for pre
    "HybridCORELSPostClassifier": {
        "build": lambda bbox, h: HybridCORELSPostClassifier(
            black_box_classifier=bbox,
            beta=h["beta"],
            c = h["lambdaValue"],
            min_coverage=h["min_coverage"],
            bb_pretrained=True,
            max_length=h["max_lenght"],
            **h["corels_params"]
        ),
        "fit": lambda model, X, y, h: model.fit(X, y, features=h["features"],
                                                                prediction_name=h['prediction_name'], time_limit=h["time_limit"],
                                                                memory_limit=h["memory_limit"], black_box_predictions=h["bb_predictions"]),
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
    'min_support':0.01, #it is changed from 0.05 to 0.01 to allow more rules to be generated for the post-hybrid model
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


   
    
    # #split information based on size
    # train_proportion=0.8

    # Load data
    my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
    
    splits = {"train": 2000, "test":2000, "validation":2000}
    X, y, features, prediction = my_data.split_data_as_dict_withsize(splits, random_state_param = seed)

    #to have bb_predictions, I would need the binarized dataset :
    #load the binarized data
    data_binarized = Dataset.from_csv(Path.cwd()/f'data/{dataset_name}.csv', dataset_name)
    X_binarized, y_binarized, _, _ = data_binarized.split_data_as_dict_withsize(splits, random_state_param = seed)
               
    start_time = time.time()
    # # Fit a black-box
    # bbox = RandomForestClassifier(random_state=42, min_samples_leaf=10, max_depth=10)
    # bbox.fit(X["train"], y["train"])

    # Retrieve the BB
    bbox_type = 'random_forest'
    model_path = Path("models") / f"{dataset_name}_{bbox_type}_{seed}.pickle"
    if not model_path.exists():
        ValueError(f"Black box model not found at {model_path}. Please ensure the black box is trained and saved before running this experiment.")

    print("Loading the Black Box")

    bbox = BlackBox(bb_type=bbox_type).load(model_path)
    bb_predictions = {k: bbox.predict(X_binarized[k]) for k in splits.keys()}
    print(f"stand alone black box train acc is {np.mean(bb_predictions['train']==y_binarized['train']):.2f}")

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
        "prediction_name": prediction, 
        "bb_predictions":bb_predictions['train']
    }




    #Build and fit the model over whole training data
    hybridmodel = spec["build"](bbox, h)
    spec["fit"](hybridmodel, X["train"], y["train"], h)
    end_time = time.time()
    solving_time = end_time - start_time

    if model in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
        rules = tuple([i['antecedents'][0]-1 for i in hybridmodel.interpretable_part.rl().rules][:-1] )#last one is the default rule, so I remove it
        status = hybridmodel.get_status()
        print("status of CORELS is : ", status)
        


    pred = {k: hybridmodel.predict_with_type(X[k],black_box_predictions=bb_predictions[k])[0]  for k in splits.keys()}
    pred_type =  {k: hybridmodel.predict_with_type(X[k],black_box_predictions=bb_predictions[k])[1]  for k in splits.keys()}
    acc = {k:float(np.mean(pred[k] == y[k])) for k in splits.keys()}
    transp_ratio = {k:float(np.mean(pred_type[k])) for k in splits.keys()}


    results = [{
    "test-train-split-seed": seed,
    "dataset_name": dataset_name,
    "model": model,
    "min_coverage": min_coverage,
    "max_lenght": max_lenght,
    "lambdaValue": lambdaValue,
    "rules": rules,
    "train":{ "predictions": pred['train'],
              "pred_type": pred_type['train'], "acc": acc['train'], "transparency": transp_ratio['train']},
    "test":{ "predictions": pred['test'],
              "pred_type": pred_type['test'], "acc": acc['test'], "transparency": transp_ratio['test']},
    "validation":{ "predictions": pred['validation'],
              "pred_type": pred_type['validation'], "acc": acc['validation'], "transparency": transp_ratio['validation']},
    "status": status,
    "solving_time": solving_time
    }]

  

    return results




LOC_PATH = Path.home()/'programming'/'optimization'/'HybridCorels-julien'
REM_PATH = "zibaja@nibi.alliancecan.ca:/home/zibaja/scratch"

def send():
    print("Sending project to Compute Canada...")
    
    cmd_str = " ".join([
        "rsync -av",
        "--exclude '.venv'",
        "--exclude '__pycache__/'",
        "--exclude '*.pyc'",
        "--exclude '.git'",
        "--exclude 'paper/plots/'", 
        "--exclude 'paper/results/'", 
        "--exclude 'paper/results_1/'", 
        "--exclude 'paper/HybridCORELS_results/'", 
        "--exclude 'paper/Mitigation_results/'", 
         "--exclude 'paper/bootstrap_results/'", 
        f"{LOC_PATH}",
        f"{REM_PATH}"
    ])
    
    subprocess.run(cmd_str, shell=True)


def receive(): 
    print("Receiving results from Compute Canada...")
    src_path = Path(REM_PATH, "HybridCorels-julien", "HybridCORELS", "paper", "HybridCORELS_results/*")
    dst_path = Path(LOC_PATH, "HybridCORELS", "paper", "HybridCORELS_results")

    cmd_str = f"rsync -av {src_path} {dst_path}"
    
    subprocess.run(cmd_str, shell=True)





def main():
    parser = argparse.ArgumentParser(description='Run bootstrap experiments for one dataset, model, and seed.')
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--local_id', type=int, default=None)
    parser.add_argument('--send', action='store_true', help='Sync project to Compute Canada')
    parser.add_argument('--receive', action='store_true', help='Fetch results from Compute Canada')

    args = parser.parse_args()


    if args.send:
        send()
        return

    if args.receive:
        receive()
        return
    

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
    print(f"Running configuration: {cfg}")
    results = run_one_model(time_limit=3600, **cfg)
   
    print(results)
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



    

if __name__ == "__main__":
    main()








