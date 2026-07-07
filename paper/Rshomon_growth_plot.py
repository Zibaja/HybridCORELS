
#March 26,2024, Version 1.0
#Rashomon set Growth 

import numpy as np
from HybridCORELS import *
from pathlib import Path
import pandas as pd
from exp_utils import *
from HyRS import HybridRuleSetClassifier
from companion_rule_list import CRL
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D



#number of models in the Rashomon set.

##############################



###############################



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
            "memory_limit": 8000,
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
            "n_iteration": 10**7, 
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
            "n_iteration": 10**7,#50000
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



###############################################
#Rashomon Set Growth , for each trade_off value, categorized in 
# Transparency groups (low, medium, high, very high) based on the trade-off parameter values
###############################################  


result_dir = Path.cwd()/"bootstrap_results"
DATASETS = ["compas", "adult", "acs_employ"]
#average coverage is computed with below function

def average_coverage (Dataset_name, method):
    seeed_split = 0

    avg_coverage = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        all_models = []
        for f in result_dir.iterdir():
            if Dataset_name in f.name and method in f.name and f"seed{seeed_split}" in f.name and f.name.endswith(f"param{tradeoff_value}.pkl"):
                with open(f, "rb") as f:
                    one_round = pickle.load(f)
                    all_models.extend(one_round)
        avg_coverage[tradeoff_value] = float(np.mean([i['coverage_rate_train'] for i in all_models]))
        #print(tradeoff_value,np.mean([i['coverage_rate_train'] for i in all_models]))
    return avg_coverage

# average_coverage('compas',"CRL" )

TRANSPARENCY_GROUPS ={
    "HybridCORELSPreClassifier": {
            "low": [0.1, 0.2, 0.3],
            "medium": [0.4, 0.5, 0.6],
            "high": [0.7, 0.8, 0.9],
            "very_high": [0.95]},
    "HybridCORELSPostClassifier": {
            "low": [0.1, 0.2, 0.3],
            "medium": [0.4, 0.5, 0.6],
            "high": [0.7, 0.8, 0.9],
            "very_high": [0.95]},
    "CRL": { # it is the opposit 
            "low":        [0.001, 0.0016681, 0.00278256],
            "medium":     [0.00464159, 0.00774264],
            "high":       [0.0129155, 0.02154435, 0.03593814,0.05994843],
            "very_high":  [ 0.1]},
    "HyRS": {
            "low":        [0.001, 0.00215443, 0.00464159, 0.01],
            "medium":     [0.02154435],
            "high":       [0.04641589],
            "very_high":  [0.1, 0.21544347, 0.46415888, 1.0]},
}




def plot_error_charts_transparency_group(Dataset_name, method):

    for group_name, group_values in TRANSPARENCY_GROUPS[method].items():
        
        plt.figure(figsize=(8,6))
        
        for tradeoff_value in TRADEOFF_VALUES[method]:
            
            # skip if not in this group
            if tradeoff_value not in group_values:
                continue
            
            # --- compute Rashomon curve ---
            _, unique_models, _ = Rashomon_set(
                Dataset_name=Dataset_name,
                method=method,
                seeed_split=0,
                tradeoff_value=tradeoff_value,
                epsilon=0.01,
                result_dir=result_dir
            )
            
            accs = np.array([m['acc_train'] for m in unique_models])
            max_acc = accs.max()
            
            eps_values = 1 - accs / max_acc
            eps_sorted = np.sort(eps_values)
            
            eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
            counts_cum = np.cumsum(counts_unique)
            
            x_vals = 1 - eps_unique
            
            # --- plot ---
            plt.step(
                x_vals,
                counts_cum,
                where='post',
                label=f"{TRADEOFF_PARAM[method]}={tradeoff_value}"
            )
        
        # --- figure formatting ---
        plt.xlabel("1-Error Tolerance (fraction of best accuracy)")
        plt.ylabel("Number of unique models in the Rashomon set")
        plt.title(f"{Dataset_name.capitalize()} | {method} | {group_name}")
        plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()
        
        # --- save ---
        output_dir = Path.cwd()/'plots'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"RS_{Dataset_name}_{method}_{group_name}.png"
        #plt.savefig(output_file, bbox_inches="tight")
        plt.show()
        plt.close()

#plot_error_charts_transparency_group("compas", "CRL")
"""Approximate Rashomon set Growth: Number of unique models in approximate Rashomon set with 
accuracy greater than or equal to (1 - ϵ) times the maximum accuracy."""


##################################################
#same plot as above only in 4 subplots

##################################################


def plot_grouped_subplots(Dataset_name, method):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax_idx, (group_name, group_values) in enumerate(TRANSPARENCY_GROUPS[method].items()):
        
        ax = axes[ax_idx]
        
        for tradeoff_value in TRADEOFF_VALUES[method]:
            
            if tradeoff_value not in group_values:
                continue
            
            # --- compute Rashomon curve ---
            _, unique_models, _ = Rashomon_set(
                Dataset_name=Dataset_name,
                method=method,
                seeed_split=0,
                tradeoff_value=tradeoff_value,
                epsilon=0.01,
                result_dir=result_dir
            )

            accs = np.array([m['acc_train'] for m in unique_models])
            max_acc = accs.max()

            eps_values = 1 - accs / max_acc
            eps_sorted = np.sort(eps_values)

            eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
            counts_cum = np.cumsum(counts_unique)

            x_vals = 1 - eps_unique

            # --- plot on subplot ---
            ax.step(
                x_vals,
                counts_cum,
                where='post',
                label=f"{TRADEOFF_PARAM[method]}={tradeoff_value}"
            )

        # --- subplot formatting ---
        ax.set_title(group_name.capitalize())
        ax.grid(True)
        ax.invert_xaxis()

        # only left plots get y-label
        if ax_idx % 2 == 0:
            ax.set_ylabel("Number of models")

        # only bottom plots get x-label
        if ax_idx >= 2:
            ax.set_xlabel("1 - ε (fraction of best accuracy)")

        ax.legend(fontsize=8)

    # --- overall title ---
    fig.suptitle(f"{Dataset_name.capitalize()} | {method}", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.gca().invert_xaxis()
    # --- save ---
    output_dir = Path.cwd() / 'plots'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"RS_{Dataset_name}_{method}_subplots.png"
    plt.show()
    #plt.savefig(output_file, bbox_inches="tight")
    plt.close()


###############################################
#Rashomon Set Growth , for each trade_off value , NO Transparency groups
###############################################


def plot_error_charts(Dataset_name,method ):
    for tradeoff_value in TRADEOFF_VALUES[method]:
        #get all the uniqe models for each trade_off value , epsilon does not matter here
        _,unique_models,_ = Rashomon_set(Dataset_name=Dataset_name, method= method, seeed_split=0,\
            tradeoff_value = tradeoff_value,epsilon=0.01,result_dir=result_dir)
        
        #print(tradeoff_value,len(unique_models) )
        accs = np.array([m['acc_train'] for m in unique_models])
        max_acc = accs.max()
        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)
        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)
        
        plt.figure(figsize=(8,6))
        plt.step(1-eps_unique, counts_cum,label = f"{TRADEOFF_PARAM[method]}={tradeoff_value}", where='post')
        #plt.plot(eps_sorted, np.arange(1, len(eps_sorted)+1))
        plt.xlabel("1-Error Tolerance $\epsilon$ (as fraction of best model)")
        plt.ylabel("Number of unique models in the Rashomon set")
        plt.title(f"{Dataset_name.capitalize()}_{method}")
        plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()
        # --- save ---
        output_dir = Path.cwd() / 'plots'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"RS_{Dataset_name}_{method}.png"
        plt.show()
        #plt.savefig(output_file, bbox_inches="tight")
        plt.close()



###############################################
#Rashomon Set Growth , based on Coverage Quntiles 
###############################################

def Rashomon_Growth_for_quantiles (all_models_per_quantiles, quantiles):
    for i,q in enumerate(all_models_per_quantiles.keys()):
        plt.figure(figsize=(8,6))
        #skip if no models in this quantile
        if len (all_models_per_quantiles[q]) == 0:
            print(f"No models in quantile {q}, skipping.")
            continue
        # --- compute Rashomon Set ---
        _, unique_models, _ = Rashomon_set_given_models(
            epsilon=0.01,
            all_models=all_models_per_quantiles[q]
        )
        print(f"Quantile: {q}, number of unique models: {len(unique_models)}")
        accs = np.array([m['acc_train'] for m in unique_models])
        max_acc = accs.max()
        
        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)
        
        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)
        
        x_vals = 1 - eps_unique
        
        # --- plot ---
        plt.step(
            x_vals,
            counts_cum,
            where='post',
            label=f"Coverage Range: {f'[{quantiles[i]:.3}, {quantiles[i+1]:.3})' if i<len(all_models_per_quantiles.keys())-1 else f'[{quantiles[i]:.3}, {quantiles[i+1]:.3}]'}"
        )

        # --- figure formatting ---
        plt.xlabel("1-Error Tolerance (fraction of best accuracy)")
        plt.ylabel("Number of unique models in the Rashomon set")
        plt.title(f"{Dataset_name.capitalize()} | {method} ")
        plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()

        # --- save ---
        output_dir = Path.cwd()/'plots'/'RS'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"RS_{Dataset_name}_{method}_{q}.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.show()
        plt.close()

def Rashomon_Growth_All_quantiles (all_models_per_quantiles, quantiles):
    n_q = len(all_models_per_quantiles)

    # Create 2x2 grid (works nicely for 4; we’ll keep it general-ish)
    n_rows = int(np.ceil(n_q / 2))
    n_cols = 2

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
    axes = axes.flatten()  # makes indexing easier

    for i, q in enumerate(all_models_per_quantiles.keys()):
        ax = axes[i]

        # --- compute Rashomon Set ---
        _, unique_models, _ = Rashomon_set_given_models(
            epsilon=0.01,
            all_models=all_models_per_quantiles[q]
        )

        print(f"Quantile: {q}, number of unique models: {len(unique_models)}")

        if len(unique_models) == 0:
            continue

        accs = np.array([m['acc_train'] for m in unique_models])
        max_acc = accs.max()

        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)

        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)

        x_vals = 1 - eps_unique

        # --- plot on subplot ---
        label = (
            f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f}]"
        )

        ax.step(x_vals, counts_cum, where='post' ) #label=label
        # --- formatting ---
        #ax.set_title(f"{q}")
        ax.set_title(f"Coverage: {label}")
        ax.set_xlabel("1 - Error Tolerance")
        ax.set_ylabel("# Rashomon Models")
        ax.grid(True)
        ax.legend()
        ax.invert_xaxis()

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Global title
    fig.suptitle(f"{Dataset_name.capitalize()} | {method}", fontsize=14)

    plt.tight_layout()
    plt.show()

    # --- save ---
    output_dir = Path.cwd() / 'plots'/'RS'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"RS_{Dataset_name}_{method}_all_quantiles.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()

def Rashomon_Growth_all_quantiles_2 (Dataset_name, method, result_dir, seed=0, n_quantiles=4, bins=None):

    # --- compute Rashomon Set ---
    epsilon_rashomon_per_quantile, unique_models_per_quantiles, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=0.01)
    n_q = len(unique_models_per_quantiles)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # makes indexing easier

    for i, q in enumerate(unique_models_per_quantiles.keys()):
        ax = axes[i]
        if len(unique_models_per_quantiles[q]) == 0:
            continue


        accs = np.array([m['acc_train'] for m in unique_models_per_quantiles[q]])
        max_acc = accs.max()

        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)

        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)

        x_vals = 1 - eps_unique

        # --- plot on subplot ---
        label = (
            f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f}]"
        )

        ax.step(x_vals, counts_cum, where='post', label = f"Coverage: {label}" ) #label=label
        # --- formatting ---
        #ax.set_title(f"{q}")
        #x.set_title(f"Coverage: {label}")
        ax.set_xlabel("1 - Error Tolerance")
        ax.set_ylabel("# Rashomon Models")
        ax.grid(True)
        ax.legend()
        ax.invert_xaxis()

    # Remove empty subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Global title
    fig.suptitle(f"{Dataset_name.capitalize()} | {method}", fontsize=14)

    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd() / 'plots'/'RS'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"RS_{Dataset_name}_{method}_all_quantiles.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()

def Rashomon_Growth_each_quantiles (Dataset_name, method, result_dir, seed=0, n_quantiles=4, bins=None):

    # --- compute Rashomon Set ---
    epsilon_rashomon_per_quantile, unique_models_per_quantiles, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=0, n_quantiles=n_quantiles, bins=bins, epsilon=0.01)
    n_q = len(unique_models_per_quantiles)
    plt.figure(figsize=(8, 6))
    for i, q in enumerate(unique_models_per_quantiles.keys()):

        if len(unique_models_per_quantiles[q]) == 0:
            continue


        accs = np.array([m['acc_train'] for m in unique_models_per_quantiles[q]])
        max_acc = accs.max()

        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)

        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)

        x_vals = 1 - eps_unique

       
        label = (
            f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f}]"
        )

        
        # --- plot ---
        plt.step(
            x_vals,
            counts_cum,
            where='post',
            label=f"Coverage: {label}")

        # --- figure formatting ---
        plt.xlabel("1-Error Tolerance")
        plt.ylabel("# Rashomon Models")
        plt.title(f"{Dataset_name.capitalize()} | {method} ")
        plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis()

        # --- save ---
        output_dir = Path.cwd()/'plots'/'RS'
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / f"RS_{Dataset_name}_{method}_{q}.png"
        plt.savefig(output_file, bbox_inches="tight")
        #plt.show()
        plt.close()


def Rashomon_Growth_all_quantiles_sameXaxis (Dataset_name, method, result_dir, seed=0, n_quantiles=4, bins=None):


    # --- compute Rashomon Set ---
    epsilon_rashomon_per_quantile, unique_models_per_quantiles, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=0.01)

    n_q = len(unique_models_per_quantiles)

    # --- create subplots with shared axes ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    # =========================
    # 1. Compute GLOBAL limits
    # =========================
    all_x = []
    all_y = []

    for q in unique_models_per_quantiles.keys():
        models = unique_models_per_quantiles[q]
        if len(models) == 0:
            continue

        accs = np.array([m['acc_train'] for m in models])
        max_acc = accs.max()

        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)

        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)

        x_vals = 1 - eps_unique

        all_x.extend(x_vals)
        all_y.extend(counts_cum)

    # fallback safety (in case everything is empty)
    if len(all_x) == 0:
        all_x = [0, 1]
    if len(all_y) == 0:
        all_y = [0, 1]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # =========================
    # 2. Plot each quantile
    # =========================
    for i, q in enumerate(unique_models_per_quantiles.keys()):
        ax = axes[i]
        models = unique_models_per_quantiles[q]

        if len(models) == 0:
            ax.set_visible(False)
            continue

        accs = np.array([m['acc_train'] for m in models])
        max_acc = accs.max()

        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)

        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)

        x_vals = 1 - eps_unique

        # Label
        label = (
            f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.3f}, {quantiles[i+1]:.3f}]"
        )

        
        # Plot
        ax.step(x_vals, counts_cum, where='post', label=f"Coverage: {label}")

        # Add vertical reference line at x = 0.990
        ax.axvline(x=0.990, linestyle='--', linewidth=1)
        # Formatting
        ax.set_xlabel("1 - Error Tolerance")
        ax.set_ylabel("# Rashomon Models")
        ax.grid(True)
        ax.legend()

        # IMPORTANT: consistent axis limits
        ax.set_xlim(x_max, x_min)  # reversed due to invert
        ax.set_ylim(y_min, y_max)

    # =========================
    # 3. Remove unused subplots
    # =========================
    for j in range(len(unique_models_per_quantiles), len(axes)):
        fig.delaxes(axes[j])

    # Cleaner ticks (optional but nice)
    for ax in axes:
        if ax.get_visible():
            ax.label_outer()

    # Global title
    fig.suptitle(f"{Dataset_name.capitalize()} | {method}", fontsize=14)

    plt.tight_layout()

    # =========================
    # 4. Save
    # =========================
    output_dir = Path.cwd() / 'plots' / 'RS'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"RS_{Dataset_name}_{method}_all_quantiles_SameAxis.png"
    #plt.show()
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


def Rashomon_Growth_one_quantiles (method, result_dir,quantile, seed=0, n_quantiles=4, bins=None):
    """
    This plots one quantile, three dataset, one method, same axis
    quantile arguemnet is the name of the requested quantile like q1, q2, q3, ....
    """

    plt.figure(figsize=(7, 4))
    for dataset in DATASETS:
        
        # --- compute Rashomon Set ---
        epsilon_rashomon_per_quantile, unique_models_per_quantiles, quantiles = generate_quantiles_Rashomon(
            dataset, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=0.01)

        n_q = len(unique_models_per_quantiles)

        q = quantile
        models = unique_models_per_quantiles[q]
        if len(models) == 0:
            continue
     

        # =========================
        # 2. Plot each quantile
        # =========================
        models = unique_models_per_quantiles[q]


        accs = np.array([m['acc_train'] for m in models])
        max_acc = accs.max()

        eps_values = 1 - accs / max_acc
        eps_sorted = np.sort(eps_values)

        eps_unique, counts_unique = np.unique(eps_sorted, return_counts=True)
        counts_cum = np.cumsum(counts_unique)

        x_vals = 1 - eps_unique

        # Label
    

        
        # Plot
        plt.step(x_vals, counts_cum, where='post', label=f"{dataset}")

        # Add vertical reference line at x = 0.990
        plt.axvline(x=0.990, linestyle='--', linewidth=1)
        # Formatting
        plt.xlabel("1 - Error Tolerance")
        plt.ylabel("# Rashomon Models")
        plt.grid(True)
        #plt.legend()

        # IMPORTANT: consistent axis limits
        plt.gca().invert_xaxis()

    i = list(epsilon_rashomon_per_quantile.keys()).index(quantile)
    label = (
            f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
    # Global title
    plt.title(f"Coverage: {label}", fontsize=14)

    plt.tight_layout()

    # =========================
    # 4. Save
    # =========================
    output_dir = Path.cwd() / 'plots' / 'RS'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"RS_{method}_{quantile}.pdf"
    
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()

DATASET_COLORS = {
    "compas": "tab:blue",
    "adult": "tab:orange",
    "acs_employ": "tab:green",
}

def save_RS_legend():

    output_dir = Path.cwd() / "plots" / "RS"
    output_dir.mkdir(exist_ok=True)

    legend_elements = [
        Line2D([0], [0], color=DATASET_COLORS[dataset], lw=2, label=dataset)
        for dataset in DATASETS
    ]

    legend_elements.append(
        Line2D([0], [0], color="black", linestyle="--", lw=1, label=r"$1-\epsilon = 0.990$")
    )

    legend_fig = plt.figure(figsize=(6, 0.5))
    legend_fig.legend(
        handles=legend_elements,
        loc="center",
        ncol=4,
        frameon=False
    )

    legend_fig.savefig(output_dir / "RS_shared_legend.pdf", bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    #plot_error_charts_transparency_group("compas", "HybridCORELSPostClassifier")
    #plot_grouped_subplots("compas", "HybridCORELSPostClassifier")
   # plot_error_charts('compas','HybridCORELSPostClassifier')
    #Analysis with Quantiles:
    Dataset_name = 'compas'
    method = 'HybridCORELSPostClassifier'
    #all_models_per_quantiles, quantiles = generate_quantiles(Dataset_name, method,result_dir, seed=0, n_quantiles=4)
    #Rashomon_Growth_for_quantiles (all_models_per_quantiles, quantiles)
    #Rashomon_Growth_All_quantiles ( all_models_per_quantiles, quantiles)
    #Second version that first generate unique models then quantile them:
    # Rashomon_Growth_all_quantiles_2 (Dataset_name, method, result_dir, seed=0, n_quantiles=4, bins=None)
    # Rashomon_Growth_each_quantiles (Dataset_name, method, result_dir, seed=0, n_quantiles=4, bins=None) #you can change n_quantiles 
    #Rashomon_Growth_all_quantiles_sameXaxis (Dataset_name, method, result_dir, seed=0, n_quantiles=4, bins=None) 

    # for dataset in DATASETS:
    #     for method in ESTIMATORS.keys():
            # Rashomon_Growth_all_quantiles_2 (dataset, method, result_dir, seed=0, n_quantiles=4, bins=None)
            #Rashomon_Growth_each_quantiles (dataset, method, result_dir, seed=0, n_quantiles=4, bins=None)
            #Rashomon_Growth_all_quantiles_sameXaxis (dataset, method, result_dir, seed=0, n_quantiles=4, bins=None) 
  
    for method in ESTIMATORS.keys():
        for quantile in ['q1', 'q2', 'q3', 'q4']: # for now I consider 4 quantiles
            Rashomon_Growth_one_quantiles (method, result_dir,quantile=quantile, seed=0, n_quantiles=4, bins=None)
    
    
    # save_RS_legend()
