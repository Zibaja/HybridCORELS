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
            "memory_limit": 8000,
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
            "memory_limit": 8000,
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

n_seeds = 10
DATASETS = ["compas", "adult", "acs_employ"]
SEEDS = list(range(n_seeds))

EXPERIMENTS = []

for dataset in DATASETS:
    for model in ESTIMATORS:
        for tardeoff_value in TRADEOFF_VALUES[model]:
            for seed in SEEDS:
                EXPERIMENTS.append({
                    "dataset": dataset,
                    "model": model,
                    TRADEOFF_PARAM[model]: tardeoff_value,
                    "seed": seed
                })

# for i,j in enumerate(EXPERIMENTS):
#     if j['dataset']=='acs_employ' and j['model']=='HybridCORELSPreClassifier':
#         print(i,j)
# print(len(EXPERIMENTS))
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
    pos_ratio_T = np.sum(label_T)/label_T.shape[0]
    pos_ratio_B = np.sum(label_B)/label_B.shape[0]
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



##############################



###############################

def run_one_seed(model_key, bbox, X, y, features, prediction_name,tradeoff_value, seed, conditions, time_limit, trans_total):

    spec = ESTIMATORS[model_key]
    h = spec["hparams"].copy()

    h.update({
        "features": features,
        "seed": seed,
        "time_limit": time_limit,
        "corels_params": CORELS_PARAMS,
        "prediction_name": prediction_name,
    })

    h[TRADEOFF_PARAM[model_key]] = tradeoff_value

    if callable(h.get("beta", None)):
        h["beta"] = h["beta"](X["train"], h["lambdaValue"])

    model = spec["build"](bbox, h)
    spec["fit"](model, X["train"], y["train"], h)

    if model_key in ['HybridCORELSPreClassifier','HybridCORELSPostClassifier']:
        status = model.get_status()
    else:
        status = None

    acc = {"train": [], "test": []}
    cov = {"train": [], "test": []}

    #  CRL returns multiple models 
    if model_key == "CRL":
        all_preds_tr, all_types_tr = model.predict_with_type_all(X["train"])
        all_preds_te, all_types_te = model.predict_with_type_all(X["test"])

        for p_tr, t_tr, p_te, t_te in zip(all_preds_tr, all_types_tr,all_preds_te, all_types_te):
            acc_train, coverage_rate_train = evaluate_one_output(X["train"], y["train"],p_tr, t_tr,"train", conditions, features, trans_total)
            acc["train"].append(acc_train)
            cov["train"].append(coverage_rate_train)

            acc_test, coverage_rate_test = evaluate_one_output(X["test"], y["test"],p_te, t_te,"test", conditions, features, trans_total)
            acc["test"].append(acc_test)
            cov["test"].append(coverage_rate_test)

    # ---- Single-model methods ----
    else:
        preds_train, preds_types_train= model.predict_with_type(X["train"])
        preds_test, preds_types_test = model.predict_with_type(X["test"])

        # label_train_T = y["train"][preds_types_train==1] #all labels going through interpretable
        # label_train_B = y["train"][preds_types_train==0] #all labels going through BB
        # pos_ration_T = np.sum(label_train_T)/label_train_T.shape[0]
        # pos_ration_B = np.sum(label_train_B)/label_train_B.shape[0]
        
        acc_train, coverage_rate_train = evaluate_one_output(X["train"], y["train"],preds_train, preds_types_train,"train", conditions, features, trans_total)
        acc["train"] = acc_train
        cov["train"] = coverage_rate_train
        


        # label_train_T = y["test"][preds_types_test==1] #all labels going through interpretable
        # label_train_B = y["test"][preds_types_test==0] #all labels going through BB
        # pos_ration_T = np.sum(label_train_T)/label_train_T.shape[0]
        # pos_ration_B = np.sum(label_train_B)/label_train_B.shape[0]
        acc_test, coverage_rate_test = evaluate_one_output(X["test"], y["test"],preds_test, preds_types_test ,"test", conditions, features, trans_total)
        acc["test"] = acc_test
        cov["test"] = coverage_rate_test


    return {
        "model": model_key,
        TRADEOFF_PARAM[model_key]: tradeoff_value,
        "accuracy": acc,
        "coverage": cov,
        "seed": seed,
        "status": status,
    }



def run_single_experiment(cfg):

    dataset_name = cfg["dataset"]
    model_key = cfg["model"]
    tradeoff_param = TRADEOFF_PARAM[model_key]
    tradeoff_value = cfg[tradeoff_param]
    seed = cfg["seed"]

    np.random.seed(seed)

    time_limit=3600 #to be changed later
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


    result_one_seed = run_one_seed( 
        model_key=model_key,
        bbox=bbox,
        X={"train":df_X["train"], "test":df_X["test"]} if model_key in ["HyRS","CRL"] else X,
        y=y,
        features=features,
        prediction_name = prediction,
        tradeoff_value=tradeoff_value,
        seed=seed,
        time_limit=time_limit, 
        conditions=condition, trans_total = trans_total)


    result = {
    "dataset": dataset_name,
    "model": model_key,
    TRADEOFF_PARAM[model_key]: tradeoff_value,
    "seed": seed,
    "accuracy": result_one_seed["accuracy"],
    "coverage": result_one_seed["coverage"],
    "trans_total": trans_total,
    "status": result_one_seed["status"]}

    out_dir = Path("results_1")
    out_dir.mkdir(exist_ok=True)

    fname = f"{dataset_name}__{model_key}__{TRADEOFF_PARAM[model_key]}_{tradeoff_value}__seed_{seed}.json"
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
    tradeoff_param = TRADEOFF_PARAM[model_key]
    tradeoff_value = cfg[tradeoff_param]
    seed = cfg["seed"]

    run_single_experiment(cfg)

   
