import numpy as np
from HybridCORELS import *
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from HybridCORELS import HybridCORELSPreClassifier, HybridCORELSPostClassifier
from exp_utils import FairnessMeasure, Dataset , save_json
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL
import argparse



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
            "min_coverage": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
            "min_coverage": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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
            "beta": 0.02,
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
            "alpha": 0.01,
            "n_iteration": 50000,
        },
    },
}

#Shared CORELS parameters
CORELS_PARAMS = {
    "policy": "objective",
    "max_card": 1,
    "n_iter": 10**7,
    'min_support':0.05,
    "verbosity": ["hybrid"],
}


# ===============================
# Experiment grid (FLATTENED)
# ===============================


n_seeds = 50
DATASETS = ["compas", "adult", "acs_employ"]
SEEDS = list(range(n_seeds))

EXPERIMENTS = []

for dataset in DATASETS:
    for model in ESTIMATORS:
        min_covs = ESTIMATORS[model]["hparams"].get("min_coverage", [None])
        for min_cov in min_covs:
            for seed in SEEDS:
                EXPERIMENTS.append({
                    "dataset": dataset,
                    "model": model,
                    "min_coverage": min_cov,
                    "seed": seed
                })

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

    icf = fairness.compute_fairness(preds_type, complement=False)['percentage_interpretable']

    CM = fairness.confusion_matrix(
        preds, y, preds_type,
        fairness.cond_indices,
        detailed=True
    )

    return icf, parse_confusion_matrix(CM)


##############################



#################################


def run_one_seed(model_key,bbox,X,y,features, prediction_name, min_cov, seed, conditions, time_limit,trans_total):

    spec = ESTIMATORS[model_key]
    h = spec["hparams"].copy()

    h.update({
        "features": features,
        "seed": seed,
        "time_limit": time_limit,
        "corels_params": CORELS_PARAMS,
        "prediction_name": prediction_name,
    })

    if min_cov is not None:
        h["min_coverage"] = min_cov

    if callable(h.get("beta", None)):
        h["beta"] = h["beta"](X["train"], h["lambdaValue"])

    model = spec["build"](bbox, h)
    spec["fit"](model, X["train"], y["train"], h)

    #for train evaluation
    preds_train, preds_types_train = model.predict_with_type(X["train"])
    acc_train = np.mean(preds_train == y["train"])
    coverage_rate_train = preds_types_train.mean()

    #for test evaluation
    preds_test, preds_types_test = model.predict_with_type(X["test"])
    acc_test = np.mean(preds_test == y["test"])
    coverage_rate_test = preds_types_test.mean()

    results = {
    "model": model_key,
    "min_coverage": min_cov,
    "accuracy": {'train': acc_train, 'test': acc_test},
    "coverage": {'train': coverage_rate_train, 'test': coverage_rate_test},
    "seed":seed}
    
    for cond in conditions:

        # ---- TRAIN ----
        icf, cm = evaluate_group(
            X["train"].to_numpy() if isinstance(X['train'], pd.DataFrame) else X["train"], y["train"],
            preds_train, preds_types_train,
            cond, features
        )


        trans_total['train'][cond]['ICF'].append(icf)

        for src in ['T', 'B']:
            for k in ['TP', 'FP', 'TN', 'FN']:
                trans_total['train'][cond][k][src].append(cm[src][k])

        # ---- TEST ----
        icf, cm = evaluate_group(
            X["test"].to_numpy() if isinstance(X['test'], pd.DataFrame) else X["test"] , y["test"],
            preds_test, preds_types_test,
            cond, features
        )

        trans_total['test'][cond]['ICF'].append(icf)
        
        for src in ['T', 'B']:
            for k in ['TP', 'FP', 'TN', 'FN']:
                trans_total['test'][cond][k][src].append(cm[src][k])

    return results


def run_single_experiment(cfg):

    dataset_name = cfg["dataset"]
    model_key = cfg["model"]
    min_cov = cfg["min_coverage"]
    seed = cfg["seed"]

    np.random.seed(seed)

    time_limit=10
    train_proportion=0.8
    # Load data
    my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
    my_data.pre_process()

    conditions = my_data.demographicGroup()
    condition = conditions['All']
    trans_total = init_trans_total(condition)
    X, y, features, prediction = my_data.get_data_norulemining(
                            {"train" : train_proportion, "test" : 1-train_proportion}, random_state_param = seed)

    #convert numpy arrays to dataframes for HyRS and CRL
    df_X = my_data.to_df_from_dict(X)

    # Fit a black-box
    bbox = RandomForestClassifier(random_state=seed, min_samples_leaf=10, max_depth=10)
    bbox.fit(df_X["train"], y["train"])

    # Set parameters


    result_one_seed = run_one_seed( ### corret it 
        model_key=model_key,
        bbox=bbox,
        X={"train":df_X["train"], "test":df_X["test"]} if model_key in ["HyRS","CRL"] else X,
        y=y,
        features=features,
        prediction_name = prediction,
        min_cov=min_cov,
        seed=seed,
        time_limit=time_limit,  #should be changed later
        conditions=condition, trans_total = trans_total)


    result = {
    "dataset": dataset_name,
    "model": model_key,
    "min_coverage": min_cov,
    "seed": seed,
    "accuracy": result_one_seed["accuracy"],
    "coverage": result_one_seed["coverage"],
    "trans_total": trans_total}

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    fname = f"{dataset_name}__{model_key}__mincov_{min_cov}__seed_{seed}.json"
    save_json(result, out_dir / fname)


# ===============================
# Entry point
# ===============================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--expe_id", type=int, required=True)
    args = parser.parse_args()
    expe_id = args.expe_id

    cfg = EXPERIMENTS[expe_id]

    dataset_name = cfg["dataset"]
    model_key = cfg["model"]
    min_cov = cfg["min_coverage"]
    seed = cfg["seed"]

    run_single_experiment(cfg)

   
