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
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from scipy.stats import norm
from matplotlib.ticker import FormatStrFormatter


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


########################################
# ICF : comprehensive chart including box plots for IC of all subgroups + IC disparity (max disparity) 
########################################
"""Box plots show the distribution of interpretability coverage across
    demographic groups for models in the Rashomon set at each hyperparameter value.
The solid line (right axis) represents the mean interpretability coverage disparity across groups,
    with shaded regions indicating one standard deviation.
"""
#Option 1 : Interpretability Coverage and Disparity Across Hyperparameters
result_dir = Path.cwd()/"bootstrap_results"
DATASETS = ["compas", "adult", "acs_employ"]
groups = ['Gender', 'Age', 'Race']
dataset_name = "compas"
method = 'HybridCORELSPreClassifier'
epsilon = 0.01 



def comprehensive_ICF_plot(dataset_name,method,epsilon,demographic_group,split):
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )

    
    conditions = my_data.demographicGroup(summarized=True)[demographic_group]
    eval = Evaluation(X[split], features, conditions)
    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------
    All_all = {cond: [] for cond in conditions}
    mean_icf_disparity = []
    std_icf_disparity = []

    for tradeoff_value in TRADEOFF_VALUES[method]:

        all_icf = {cond: [] for cond in conditions}
        all_icf_disparity = []

        for model in results[tradeoff_value]:
            ICF = eval.compute_fairness(model[f'preds_types_{split}'])
            ICF_disparity = eval.compute_ICF_disparity(ICF)

            all_icf_disparity.append(ICF_disparity['over_all_groups'])

            for cond in conditions:
                all_icf[cond].append(ICF[cond])

        mean_icf_disparity.append(np.mean(all_icf_disparity))
        std_icf_disparity.append(np.std(all_icf_disparity))

        for cond in conditions:
            All_all[cond].append(all_icf[cond])

    # -------------------------------
    # PLOTTING (UPDATED FOR ANY #GROUPS)
    # -------------------------------
    plt.figure(figsize=(10, 6))

    num_groups = len(conditions)
    num_tradeoffs = len(TRADEOFF_VALUES[method])

    base_positions = np.arange(1, num_tradeoffs + 1)

    # dynamic spacing + width
    total_width = 0.8
    width = total_width / num_groups

    # dynamic colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    # --- boxplots ---
    for i, cond in enumerate(conditions):

        # center groups around base position
        positions = base_positions + (i - (num_groups - 1)/2) * width

        box = plt.boxplot(
            All_all[cond],
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])

    # --- disparity line ---
    plt.plot(
        base_positions,
        mean_icf_disparity,
        color='black',
        marker='o',
        linewidth=2
    )

    plt.fill_between(
        base_positions,
        np.array(mean_icf_disparity) - np.array(std_icf_disparity),
        np.array(mean_icf_disparity) + np.array(std_icf_disparity),
        color='gray',
        alpha=0.25
    )

    # -------------------------------
    # Legend
    # -------------------------------
    legend_elements = [
        Patch(facecolor=colors[i], label=cond)
        for i, cond in enumerate(conditions)
    ]

    legend_elements.append(
        Line2D([0], [0], color='black', marker='o', label='IC Disparity')
    )

    plt.legend(handles=legend_elements)

    # -------------------------------
    # Labels & formatting
    # -------------------------------
    plt.xticks(base_positions, TRADEOFF_VALUES[method], rotation=45)

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Interpretability Coverage')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Comp_ICF_{dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def comprehensive_ICF_plot_two_axis(dataset_name,method,epsilon,demographic_group,split):
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )

    
    conditions = my_data.demographicGroup(summarized=True)[demographic_group]
    eval = Evaluation(X[split], features, conditions)
    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------
    All_all = {cond: [] for cond in conditions}
    mean_icf_disparity = []
    std_icf_disparity = []

    for tradeoff_value in TRADEOFF_VALUES[method]:

        all_icf = {cond: [] for cond in conditions}
        all_icf_disparity = []

        for model in results[tradeoff_value]:
            ICF = eval.compute_fairness(model[f'preds_types_{split}'])
            ICF_disparity = eval.compute_ICF_disparity(ICF)

            all_icf_disparity.append(ICF_disparity['over_all_groups'])

            for cond in conditions:
                all_icf[cond].append(ICF[cond])

        mean_icf_disparity.append(np.mean(all_icf_disparity))
        std_icf_disparity.append(np.std(all_icf_disparity))

        for cond in conditions:
            All_all[cond].append(all_icf[cond])

    # -------------------------------
    # PLOTTING (UPDATED FOR ANY #GROUPS)
    # -------------------------------
        
    # Using two y axes due to scale difference
    fig, ax1 = plt.subplots(figsize=(10, 6))

    num_groups = len(conditions)
    num_tradeoffs = len(TRADEOFF_VALUES[method])

    base_positions = np.arange(1, num_tradeoffs + 1)

    # dynamic spacing
    total_width = 0.8
    width = total_width / num_groups

    # dynamic colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    # --- BOX PLOTS (left axis) ---
    for i, cond in enumerate(conditions):

        # center all groups
        positions = base_positions + (i - (num_groups - 1)/2) * width

        box = ax1.boxplot(
            All_all[cond],
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])

    ax1.set_ylabel('Interpretability Coverage')
    ax1.set_xlabel(f'{TRADEOFF_PARAM[method]}')

    # --- SECOND AXIS (right) ---
    ax2 = ax1.twinx()

    ax2.plot(
        base_positions,
        mean_icf_disparity,
        color='black',
        marker='o',
        linewidth=2,
        label='IC Disparity'
    )

    ax2.fill_between(
        base_positions,
        np.array(mean_icf_disparity) - np.array(std_icf_disparity),
        np.array(mean_icf_disparity) + np.array(std_icf_disparity),
        color='gray',
        alpha=0.2
    )

    ax2.set_ylabel('Interpretability Coverage Disparity')

    # --- LEGENDS ---
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # boxplot legend (groups)
    legend_elements = [
        Patch(facecolor=colors[i], label=cond)
        for i, cond in enumerate(conditions)
    ]

    ax1.legend(handles=legend_elements, loc='upper left')

    # disparity legend (line)
    ax2.legend(loc='upper right')

    # --- formatting ---
    plt.title(f'{dataset_name.capitalize()} | {method}')
    ax1.set_xticks(base_positions)
    ax1.set_xticklabels(TRADEOFF_VALUES[method], rotation=45)

    ax1.grid(axis='y')
    plt.tight_layout()
    # # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Comp_ICF_2axis{dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def max_delta_ICF (dataset_name,method,epsilon,split):
    # -------------------------------
    # Data preparation (unchanged)
    # -------------------------------
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)

    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set


    # -------------------------------
    # Compute statistics
    # -------------------------------
    plt.figure(figsize=(7, 4))
    for g in groups:
        demographic_group = g
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_icf_disparity = []
        std_icf_disparity = []

        for tradeoff_value in TRADEOFF_VALUES[method]:

            
            all_icf_disparity = []

            for model in results[tradeoff_value]:
                ICF = eval.compute_fairness(model[f'preds_types_{split}'])
                ICF_disparity = eval.compute_ICF_disparity(ICF)

                all_icf_disparity.append(ICF_disparity['over_all_groups'])


            mean_icf_disparity.append(np.mean(all_icf_disparity))
            std_icf_disparity.append(np.std(all_icf_disparity))
        #if i want to use normalized values 
        # x_vals = np.array(TRADEOFF_VALUES[method])
        # x_norm = x_vals / x_vals.max()
        plt.plot(
            TRADEOFF_VALUES[method],
            mean_icf_disparity,
            label = f'Maximum IC disparity-{g}',
            linewidth=2
        )

        plt.fill_between(
            TRADEOFF_VALUES[method],
            np.array(mean_icf_disparity) - np.array(std_icf_disparity),
            np.array(mean_icf_disparity) + np.array(std_icf_disparity),
            color='gray',
            alpha=0.25
        )


    plt.legend()
    # as for CRL and HyRS the x_axis has logarithmically spaced values
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Interpretability Coverage Disparity')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Max_Delta_ICF{dataset_name}_{method}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def delta_ICF_for_pairs (dataset_name,method,epsilon,split,demographic_group):

    # -------------------------------
    # Data preparation 
    # -------------------------------
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------


    conditions = my_data.demographicGroup(summarized=True)[demographic_group]

    eval = Evaluation(X[split], features, conditions)


    mean_icf_disparity = {pairs: [] for pairs in paired_subgroups(conditions)}
    std_icf_disparity = {pairs: [] for pairs in paired_subgroups(conditions)}
    plt.figure(figsize=(7, 4))
    for tradeoff_value in TRADEOFF_VALUES[method]:

        
        all_icf_disparity_paired = {pairs: [] for pairs in paired_subgroups(conditions)}

        for model in results[tradeoff_value]:
            ICF = eval.compute_fairness(model[f'preds_types_{split}'])
            ICF_disparity = eval.compute_ICF_disparity(ICF)
            
            for pairs in paired_subgroups(conditions):
                all_icf_disparity_paired[pairs].append(ICF_disparity[pairs])

        for pairs in paired_subgroups(conditions):
            mean_icf_disparity[pairs].append(float(np.mean(all_icf_disparity_paired[pairs])))
            std_icf_disparity[pairs].append(float(np.std(all_icf_disparity_paired[pairs])))


    #print(mean_icf_disparity[('Gender=Male', 'neg_Gender=Male')])
    #if i want to use normalized values 
    # x_vals = np.array(TRADEOFF_VALUES[method])
    # x_norm = x_vals / x_vals.max()

    for pairs in paired_subgroups(conditions):

        plt.plot(
            TRADEOFF_VALUES[method],
            mean_icf_disparity[pairs],
            label = f'IC disparity: ({pairs[0]})-({pairs[1]})',
            linewidth=2
        )

        plt.fill_between(
            TRADEOFF_VALUES[method],
            np.array(mean_icf_disparity[pairs]) - np.array(std_icf_disparity[pairs]),
            np.array(mean_icf_disparity[pairs]) + np.array(std_icf_disparity[pairs]),
            color='gray',
            alpha=0.25
        )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Interpretability Coverage Disparity')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Delta_ICF_pairs{dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def max_EO (dataset_name,method,epsilon,split, model_part):


    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------
    plt.figure(figsize=(7, 4))
    
    for g in groups:
        demographic_group = g
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_EO = []
        std_EO = []

        for tradeoff_value in TRADEOFF_VALUES[method]:

            all_EO = []

            for model in results[tradeoff_value]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                TPR = eval.compute_true_pos_ratio(CM)
                EO = eval.compute_Equal_Opportunity(TPR, model_part=model_part)
                all_EO.append(EO['over_all_groups'])


            mean_EO.append(float(np.mean(all_EO)))
            std_EO.append(float(np.std(all_EO)))
        #if i want to use normalized values 
        # x_vals = np.array(TRADEOFF_VALUES[method])
        # x_norm = x_vals / x_vals.max()
        plt.plot(
            TRADEOFF_VALUES[method],
            mean_EO,
            label = f'Maximum EO-{g}',
            linewidth=2
        )

        plt.fill_between(
            TRADEOFF_VALUES[method],
            np.array(mean_EO) - np.array(std_EO),
            np.array(mean_EO) + np.array(std_EO),
            color='gray',
            alpha=0.25
        )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Equal Opportunity')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'EO'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Max_EO{dataset_name}_{method}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def EO_for_pairs (dataset_name,method,epsilon,split,demographic_group, model_part):

    # -------------------------------
    # Data preparation
    # -------------------------------
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------


    conditions = my_data.demographicGroup(summarized=True)[demographic_group]

    eval = Evaluation(X[split], features, conditions)


    mean_EO = {pairs: [] for pairs in paired_subgroups(conditions)}
    std_EO = {pairs: [] for pairs in paired_subgroups(conditions)}
    plt.figure(figsize=(7, 4))
    for tradeoff_value in TRADEOFF_VALUES[method]:

        
        all_EO_paired = {pairs: [] for pairs in paired_subgroups(conditions)}

        for model in results[tradeoff_value]:
            CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
            TPR = eval.compute_true_pos_ratio(CM)
            EO = eval.compute_Equal_Opportunity(TPR, model_part=model_part)
            for pairs in paired_subgroups(conditions):
                all_EO_paired[pairs].append(EO[pairs])

        for pairs in paired_subgroups(conditions):
            mean_EO[pairs].append(float(np.mean(all_EO_paired[pairs])))
            std_EO[pairs].append(float(np.std(all_EO_paired[pairs])))


    #print(mean_icf_disparity[('Gender=Male', 'neg_Gender=Male')])
    #if i want to use normalized values 
    # x_vals = np.array(TRADEOFF_VALUES[method])
    # x_norm = x_vals / x_vals.max()

    for pairs in paired_subgroups(conditions):

        plt.plot(
            TRADEOFF_VALUES[method],
            mean_EO[pairs],
            label = f'EO: ({pairs[0]})-({pairs[1]})',
            linewidth=2
        )

        plt.fill_between(
            TRADEOFF_VALUES[method],
            np.array(mean_EO[pairs]) - np.array(std_EO[pairs]),
            np.array(mean_EO[pairs]) + np.array(std_EO[pairs]),
            color='gray',
            alpha=0.25
        )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Equal Opportunity')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'EO'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"EO_pairs{dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def max_SP (dataset_name,method,epsilon,split, model_part):
 
    
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------
    plt.figure(figsize=(7, 4))

    for g in groups:
        demographic_group = g
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_SP = []
        std_SP = []

        for tradeoff_value in TRADEOFF_VALUES[method]:

            all_SP = []

            for model in results[tradeoff_value]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                PPR = eval.compute_pred_pos_ratio(CM)
                SP = eval.compute_Statistical_Parity(PPR, model_part=model_part)
                all_SP.append(SP['over_all_groups'])


            mean_SP.append(float(np.mean(all_SP)))
            std_SP.append(float(np.std(all_SP)))
        #if i want to use normalized values 
        # x_vals = np.array(TRADEOFF_VALUES[method])
        # x_norm = x_vals / x_vals.max()
        plt.plot(
            TRADEOFF_VALUES[method],
            mean_SP,
            label = f'Maximum SP-{g}',
            linewidth=2
        )

        plt.fill_between(
            TRADEOFF_VALUES[method],
            np.array(mean_SP) - np.array(std_SP),
            np.array(mean_SP) + np.array(std_SP),
            color='gray',
            alpha=0.25
        )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Statistical Parity')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'SP'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Max_SP{dataset_name}_{method}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def SP_for_pairs(dataset_name,method,epsilon,split,demographic_group, model_part):


    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------


    conditions = my_data.demographicGroup(summarized=True)[demographic_group]

    eval = Evaluation(X[split], features, conditions)


    mean_SP = {pairs: [] for pairs in paired_subgroups(conditions)}
    std_SP = {pairs: [] for pairs in paired_subgroups(conditions)}
    plt.figure(figsize=(7, 4))
    for tradeoff_value in TRADEOFF_VALUES[method]:

        
        all_SP_paired = {pairs: [] for pairs in paired_subgroups(conditions)}

        for model in results[tradeoff_value]:
            CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
            PPR = eval.compute_pred_pos_ratio(CM)
            SP = eval.compute_Statistical_Parity(PPR, model_part=model_part)
            for pairs in paired_subgroups(conditions):
                all_SP_paired[pairs].append(SP[pairs])

        for pairs in paired_subgroups(conditions):
            mean_SP[pairs].append(float(np.mean(all_SP_paired[pairs])))
            std_SP[pairs].append(float(np.std(all_SP_paired[pairs])))


    #print(mean_icf_disparity[('Gender=Male', 'neg_Gender=Male')])
    #if i want to use normalized values 
    # x_vals = np.array(TRADEOFF_VALUES[method])
    # x_norm = x_vals / x_vals.max()

    for pairs in paired_subgroups(conditions):

        plt.plot(
            TRADEOFF_VALUES[method],
            mean_SP[pairs],
            label = f'SP: ({pairs[0]})-({pairs[1]})',
            linewidth=2
        )

        plt.fill_between(
            TRADEOFF_VALUES[method],
            np.array(mean_SP[pairs]) - np.array(std_SP[pairs]),
            np.array(mean_SP[pairs]) + np.array(std_SP[pairs]),
            color='gray',
            alpha=0.25
        )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Statistical Parity')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    output_dir = Path.cwd()/'plots'/'SP'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"SP_pairs{dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()


def individual_arbitrariness(dataset_name,method, epsilon,split):

    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------

    arbitrariness_mean = []
    arbitrariness_std= []
    all_arbitrariness = []
    for tradeoff_value in TRADEOFF_VALUES[method]:
        pred_types_list = []
        for model in results[tradeoff_value]:
            pred_types_list.append(model[f'preds_types_{split}'])
        all_models_pred_type = np.column_stack(pred_types_list)
        arbitrarines = np.mean(all_models_pred_type, axis=1)
        all_arbitrariness.append(arbitrarines) # all arbitrariness values for all trade_off values
        #average and std over all tradeoff values 
        arbitrariness_mean.append(float(np.mean(arbitrarines)))
        arbitrariness_std.append(float(np.std(arbitrarines)))

    #all vectors for all tradeoff values 
    average_all = np.mean(np.column_stack(all_arbitrariness), axis=1)


    plt.plot(
        TRADEOFF_VALUES[method],
        arbitrariness_mean,
        linewidth=2
    )

    plt.fill_between(
        TRADEOFF_VALUES[method],
        np.array(arbitrariness_mean) - np.array(arbitrariness_std),
        np.array(arbitrariness_mean) + np.array(arbitrariness_std),
        color='gray',
        alpha=0.25
    )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Average Individual Arbitrariness')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()

    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"Individual_arbitrariness_{dataset_name}_{method}.png"
    plt.savefig(output_file, bbox_inches="tight")
    
    plt.show()
    plt.close()

    return average_all
        
    
def group_arbitrariness (dataset_name,method,epsilon,split,demographic_group):


    # -------------------------------
    # Data preparation
    # -------------------------------
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------

    conditions = my_data.demographicGroup(summarized=True)[demographic_group]
    eval = Evaluation(X[split], features, conditions)

    arbitrarines_all = {cond:[] for cond in conditions}
    arbitrariness_mean = []
    arbitrariness_std= []
    for tradeoff_value in TRADEOFF_VALUES[method]:
        pred_types_list = []
        for model in results[tradeoff_value]:
            pred_types_list.append(model[f'preds_types_{split}'])
        all_models_pred_type = np.column_stack(pred_types_list)
        arbitrarines = np.mean(all_models_pred_type, axis=1)
        arbitrariness_mean.append (float(np.mean(arbitrarines)))
        arbitrariness_std.append (float(np.std(arbitrarines)))
        #print(tradeoff_value,float(np.mean(arbitrarines)),float(np.std(arbitrarines)) , np.max(arbitrarines))
        for index, cond in enumerate(conditions):
            arbitrarines_all[cond].append(arbitrarines[eval.cond_indices[:,index]])

    plt.figure(figsize=(10, 6))

    num_groups = len(conditions)
    num_tradeoffs = len(TRADEOFF_VALUES[method])

    base_positions = np.arange(1, num_tradeoffs + 1)

    # dynamic spacing + width
    total_width = 0.8
    width = total_width / num_groups

    # dynamic colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    # --- boxplots ---
    for i, cond in enumerate(conditions):

        # center groups around base position
        positions = base_positions + (i - (num_groups - 1)/2) * width

        box = plt.boxplot(
            arbitrarines_all[cond],
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])




    # --- Average Individual Arbitrariness line ---
    plt.plot(
        base_positions,
        arbitrariness_mean,
        color='black',
        marker='o',
        linewidth=2
    )

    plt.fill_between(
        base_positions,
        np.array(arbitrariness_mean) - np.array(arbitrariness_std),
        np.array(arbitrariness_mean) + np.array(arbitrariness_std),
        color='gray',
        alpha=0.25
    )

    # -------------------------------
    # Legend
    # -------------------------------
    legend_elements = [
        Patch(facecolor=colors[i], label=cond)
        for i, cond in enumerate(conditions)
    ]

    legend_elements.append(
        Line2D([0], [0], color='black', marker='o', label='Individual Arbitrariness')
    )

    plt.legend(handles=legend_elements)

    # -------------------------------
    # Labels & formatting
    # -------------------------------
    plt.xticks(base_positions, TRADEOFF_VALUES[method], rotation=45)

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Arbitrariness')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid(axis='y')
    plt.tight_layout()
    #--- save ---
    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Comp_Arbitrariness_{dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()

    
def sparsity (dataset_name,method,epsilon):
   

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{dataset_name}_mined.csv',
        dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )



    # -------------------------------
    # Collect Rashomon models
    # -------------------------------
    results = {}
    for tradeoff_value in TRADEOFF_VALUES[method]:
        rashomon_set, _, _ = Rashomon_set(
            Dataset_name=dataset_name,
            method=method,
            seeed_split=0,
            tradeoff_value=tradeoff_value,
            epsilon=epsilon,
            result_dir=result_dir
        )
        results[tradeoff_value] = rashomon_set

    # -------------------------------
    # Compute statistics
    # -------------------------------


    sparsity_mean = []
    sparsity_std = []
    for tradeoff_value in TRADEOFF_VALUES[method]:
        sparsity_all = []
        for model in results[tradeoff_value]:
            if method!='HyRS':
                sparsity_all.append(len(model['rules'])) #for now only for HybridCOREL
            else:
                pos,neg = model['rules']
                sparsity_all.append(len(pos)+len(neg))

        sparsity_mean.append(np.mean(sparsity_all))
        sparsity_std.append(np.std(sparsity_all))




    plt.plot(
        TRADEOFF_VALUES[method],
        sparsity_mean,
        linewidth=2
    )

    plt.fill_between(
        TRADEOFF_VALUES[method],
        np.array(sparsity_mean) - np.array(sparsity_std),
        np.array(sparsity_mean) + np.array(sparsity_std),
        color='gray',
        alpha=0.25
    )


    plt.legend()
    if method in ['CRL','HyRS']:
        plt.xscale('log')
        import matplotlib.ticker as mticker
        plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter('%.3g'))

    plt.xlabel(f'{TRADEOFF_PARAM[method]}')
    plt.ylabel('Number of Rules')
    plt.title(f'{dataset_name.capitalize()} | {method}')

    plt.grid()
    plt.tight_layout()
    #--- save ---
    output_dir = Path.cwd()/'plots'/'Sparsity'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Sparsity_{dataset_name}_{method}.png"
    plt.savefig(output_file, bbox_inches="tight")

    plt.show()
    plt.close()

        
    
################################
#Analysis over Rashomon sets of Quantiles 
################################



def comprehensive_ICF_plot_2_axis_quantiles(Dataset_name,method,seed, result_dir, n_quantiles ,epsilon,bins,split, demographic_group):
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )


    conditions = my_data.demographicGroup(summarized=True)[demographic_group]
    eval = Evaluation(X[split], features, conditions)



    # --- compute Rashomon Set ---
    epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
    Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
    )

    # -------------------------------
    # Compute statistics
    # -------------------------------
    All_all = {cond: [] for cond in conditions}
    mean_icf_disparity = []
    std_icf_disparity = []
    for q in epsilon_rashomon_per_quantile.keys():

        all_icf = {cond: [] for cond in conditions}
        all_icf_disparity = []

        for model in epsilon_rashomon_per_quantile[q]:
            ICF = eval.compute_fairness(model[f'preds_types_{split}'])
            ICF_disparity = eval.compute_ICF_disparity(ICF)

            all_icf_disparity.append(ICF_disparity['over_all_groups'])

            for cond in conditions:
                all_icf[cond].append(ICF[cond])

        mean_icf_disparity.append(np.mean(all_icf_disparity))
        std_icf_disparity.append(np.std(all_icf_disparity))

        for cond in conditions:
            All_all[cond].append(all_icf[cond])

    # -------------------------------
    # PLOTTING (UPDATED FOR ANY #GROUPS)
    # -------------------------------
        
    # Using two y axes due to scale difference
    fig, ax1 = plt.subplots(figsize=(10, 6))

    LABEL_SIZE = 20
    TICK_SIZE = 17
    LEGEND_SIZE = 14
    TITLE_SIZE = 18
    
    num_groups = len(conditions)
    num_quantiles = len(epsilon_rashomon_per_quantile.keys())

    base_positions = np.arange(1, num_quantiles + 1)

    # dynamic spacing
    total_width = 0.8
    width = total_width / num_groups

    # dynamic colors
    colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

    # --- BOX PLOTS (left axis) ---
    for i, cond in enumerate(conditions):

        # center all groups
        positions = base_positions + (i - (num_groups - 1)/2) * width

        box = ax1.boxplot(
            All_all[cond],
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(colors[i])

   
    ax1.set_ylabel(
        'Interpretability Coverage',
        fontsize=LABEL_SIZE)

    ax1.set_xlabel(
        'Transparency bins',
        fontsize=LABEL_SIZE)
    # --- SECOND AXIS (right) ---

    ax2 = ax1.twinx()

    line_color = "red"
    shade_color = "lightcoral"

    ax2.plot(
        base_positions,
        mean_icf_disparity,
        color=line_color,
        marker='o',
        linewidth=2.5,
        label='ICD'
    )

    ax2.fill_between(
        base_positions,
        np.array(mean_icf_disparity) - np.array(std_icf_disparity),
        np.array(mean_icf_disparity) + np.array(std_icf_disparity),
        color=shade_color,
        alpha=0.25)

    
    ax2.set_ylabel(
    'ICD',
    color=line_color,
    fontsize=LABEL_SIZE)

    # make right axis red
    ax1.tick_params(axis='both', labelsize=TICK_SIZE)
    ax2.tick_params(axis='y', labelsize=TICK_SIZE, colors=line_color)

    ax2.spines['right'].set_color(line_color)
    ax2.yaxis.label.set_color(line_color)
    ax2.set_ylim(bottom=0)
    ax1.set_ylim(bottom=0)
    # disparity legend (line)
    ax2.legend(loc='upper right', fontsize = LEGEND_SIZE)



    if demographic_group == 'Gender':
        # boxplot legend (groups)
        legend_elements = [
            Patch(facecolor=colors[i], label='Male' if cond in ['gender_Male','Gender=Male', 'neg_Female'] else 'Female')
            for i, cond in enumerate(conditions)
        ]
    else:
        # boxplot legend (groups)
        legend_elements = [
            Patch(facecolor=colors[i], label=cond.capitalize())
            for i, cond in enumerate(conditions)
        ]

    ax1.legend(handles=legend_elements, loc='upper left', fontsize = LEGEND_SIZE)


    # --- formatting ---
    #plt.title(f'{Dataset_name.capitalize()} | {method}')
    ax1.set_xticks(base_positions)

    #to generate quantiles range labels
    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

 
    ax1.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax1.grid(axis='y')
    plt.tight_layout()
    # # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Comp_ICF_2axis{Dataset_name}_{method}_{demographic_group}_{split}_{epsilon}.pdf"
    
    plt.savefig(
    output_file,
    bbox_inches="tight",
    dpi=300)
    #plt.show()
    plt.close()


def max_delta_ICF_all_methods(Dataset_name,seed, result_dir, epsilon,split, demographic_group, n_quantiles, bins):
    """This function generates a plot comparing the maximum Interpretability Coverage Disparity 
    across different methods for a given dataset and demographic group. 
    It computes the Rashomon sets for each method, 
    calculates the ICF disparity for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the maximum ICF disparity across
      quantiles for each method on the same graph.

    """

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange',
    'HyRS': 'tab:green',
    'CRL': 'tab:red'
    }
    plt.figure(figsize=(7, 4))
    for method in ESTIMATORS:
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_icf_disparity = []
        std_icf_disparity = []

        for q in epsilon_rashomon_per_quantile.keys():

            all_icf_disparity = []

            for model in epsilon_rashomon_per_quantile[q]:
                ICF = eval.compute_fairness(model[f'preds_types_{split}'])
                ICF_disparity = eval.compute_ICF_disparity(ICF)

                all_icf_disparity.append(ICF_disparity['over_all_groups'])


            mean_icf_disparity.append(np.mean(all_icf_disparity))
            std_icf_disparity.append(np.std(all_icf_disparity))

        x_pos = np.arange(len(epsilon_rashomon_per_quantile))
        plt.plot(
            x_pos,
            mean_icf_disparity,
            label = f'{method}',
            linewidth=2, color = color_map[method]
        )

        plt.fill_between(
            x_pos,
            np.array(mean_icf_disparity) - np.array(std_icf_disparity),
            np.array(mean_icf_disparity) + np.array(std_icf_disparity),
            color='gray',
            alpha=0.25
        )


    plt.legend()

    plt.xlabel(f'Quantiles of Coverage Rate')
    plt.ylabel('Max. Interpretability Coverage Disparity')
    #plt.title(f'{Dataset_name.capitalize()}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

    plt.xticks(x_pos, labels)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_Delta_ICF{Dataset_name}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def max_delat_ICF_all_methods_withfairs (Dataset_name,seed, result_dir, epsilon,split, demographic_group, n_quantiles, bins):
    
    """the same plots as max_delta_ICF_all_methods but with colored shaded regions and fair methods
    """
    
    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results",
                'HyRS':Path.cwd()/"bootstrap_results",
                'CRL':Path.cwd()/"bootstrap_results" }
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                'HybridCORELSPost':'HybridCORELSPostClassifier',
                'HyRS': 'HyRS',
                'CRL': 'CRL'}

    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange',
    'HyRS': 'tab:green',
    'CRL': 'tab:red',

    }


    plt.figure(figsize=(7, 4))
    for method in method_match:
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[method], result_dir = result_dir[method], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_icf_disparity = []
        std_icf_disparity = []

        for q in epsilon_rashomon_per_quantile.keys():

            all_icf_disparity = []

            for model in epsilon_rashomon_per_quantile[q]:
                ICF = eval.compute_fairness(model[f'preds_types_{split}'])
                ICF_disparity = eval.compute_ICF_disparity(ICF)

                all_icf_disparity.append(ICF_disparity['over_all_groups'])


            mean_icf_disparity.append(np.mean(all_icf_disparity))
            std_icf_disparity.append(np.std(all_icf_disparity))

        x_pos = np.arange(len(epsilon_rashomon_per_quantile))
        linestyle = '--' if "Fair" in method else '-'
        plt.plot(
            x_pos,
            mean_icf_disparity,
            label = f'{method}' if "Fair" not in method else None,
            linewidth=2, color = color_map [method_match[method]], linestyle = linestyle
        )

        plt.fill_between(
            x_pos,
            np.array(mean_icf_disparity) - np.array(std_icf_disparity),
            np.array(mean_icf_disparity) + np.array(std_icf_disparity),
            color= color_map [method_match[method]],
            alpha=0.25
        )


    plt.legend() 

    plt.xlabel(f'Quantiles of Coverage Rate')
    plt.ylabel('Max. Interpretability Coverage Disparity')
    plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

    plt.xticks(x_pos, labels)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_Delta_ICF{Dataset_name}_{demographic_group}_{epsilon}.png"

    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()

#the one used in paper
def max_delta_ICF_all_methods_box(Dataset_name,seed, result_dir, epsilon,split, demographic_group, n_quantiles, bins):
    """This function generates a plot comparing the maximum Interpretability Coverage Disparity 
    across different methods for a given dataset and demographic group. 
    It computes the Rashomon sets for each method, 
    calculates the ICF disparity for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the maximum ICF disparity across
      quantiles for each method on the same graph.

    """

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange',
    'HyRS': 'tab:green',
    'CRL': 'tab:red'
    }
    
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24
    for i,method in enumerate(ESTIMATORS.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


    
        all_icf_disparity_per_quantile = []

        for q in epsilon_rashomon_per_quantile.keys():

            all_icf_disparity = []

            for model in epsilon_rashomon_per_quantile[q]:
                ICF = eval.compute_fairness(model[f'preds_types_{split}'])
                ICF_disparity = eval.compute_ICF_disparity(ICF)

                all_icf_disparity.append(ICF_disparity['over_all_groups'])

            all_icf_disparity_per_quantile.append(all_icf_disparity)
          


        num_methods = len(ESTIMATORS)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods


        # --- BOX PLOTS ---
        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_icf_disparity_per_quantile,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[method])
            patch.set_alpha(0.7)



    #plt.legend()

    plt.xlabel(f'Transparency bins', fontsize= LABEL_SIZE)
    plt.ylabel('ICD', fontsize= LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
     # --- LEGENDS ---
    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(ESTIMATORS)
    # ]

    #plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    # force 1-digit float format
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_Delta_ICF_Box_{Dataset_name}_{demographic_group}_{split}_{epsilon}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()

#legend for delta_ICF_all_methods_Box
def save_methods_legend():
    color_map = {
        'HybridCORELSPreClassifier': 'tab:blue',
        'HybridCORELSPostClassifier': 'tab:orange',
        'HyRS': 'tab:green',
        'CRL': 'tab:red'
    }

    label_map = {
        'HybridCORELSPreClassifier': 'HybridCORELSPre',
        'HybridCORELSPostClassifier': 'HybridCORELSPost',
        'HyRS': 'HyRS',
        'CRL': 'CRL'
    }

    legend_elements = [
        Line2D(
            [0], [0],
            color=color_map[method],
            lw=2,
            label=label_map[method]
        )
        for method in color_map
    ]

    legend_fig = plt.figure(figsize=(7, 0.5))

    legend_fig.legend(
        handles=legend_elements,
        loc='center',
        ncol=4,
        frameon=False
    )

    output_dir = Path.cwd() / 'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    legend_fig.savefig(
        output_dir / 'allmethods_ICF_shared_legend.pdf',
        bbox_inches='tight'
    )

    plt.close(legend_fig)


def max_EO_all_methods(Dataset_name,seed, result_dir, epsilon,split, demographic_group, model_part, n_quantiles, bins):
    """This function generates a plot comparing the maximum Equal Opportunity 
    across different methods for a given dataset and demographic group. 
    It computes the Rashomon sets for each method, 
    calculates the EO for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the maximum EO across
      quantiles for each method on the same graph.

    """
    
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange',
    'HyRS': 'tab:green',
    'CRL': 'tab:red'
    }
    plt.figure(figsize=(7, 4))
    for method in ESTIMATORS:
        
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        
        # -------------------------------
        # Compute statistics
        # -------------------------------


        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_EO = []
        std_EO = []

        for q in epsilon_rashomon_per_quantile.keys():

            all_EO = []

            for model in epsilon_rashomon_per_quantile[q]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                TPR = eval.compute_true_pos_ratio(CM)
                EO = eval.compute_Equal_Opportunity(TPR, model_part=model_part)
                all_EO.append(EO['over_all_groups'])


            mean_EO.append(float(np.mean(all_EO)))
            std_EO.append(float(np.std(all_EO)))
        
        x_pos = np.arange(len(epsilon_rashomon_per_quantile.keys()))

        plt.plot(
            x_pos,
            mean_EO,
            label = f'{method}',
            linewidth=2, color = color_map[method]
        )

        plt.fill_between(
            x_pos,
            np.array(mean_EO) - np.array(std_EO),
            np.array(mean_EO) + np.array(std_EO),
            color='gray',
            alpha=0.25
        )

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

    plt.xticks(x_pos, labels)
    plt.legend()
    plt.xlabel(f'Quantiles of Coverage Rate')
    plt.ylabel('Max. Equal Opportunity')
    plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'EO'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_EO{Dataset_name}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def max_SP_all_methods(Dataset_name,seed, result_dir, epsilon,split, demographic_group, model_part, n_quantiles, bins):
    """This function generates a plot comparing the maximum Statistical Parity (SP) 
    across different methods for a given dataset and demographic group. 
    It computes the Rashomon sets for each method, 
    calculates the EO for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the maximum EO across
      quantiles for each method on the same graph.

    """
    
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange',
    'HyRS': 'tab:green',
    'CRL': 'tab:red'
    }
    plt.figure(figsize=(7, 4))
    for method in ESTIMATORS:
        
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # -------------------------------
        # Compute statistics
        # -------------------------------


        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        mean_SP = []
        std_SP = []

        for q in epsilon_rashomon_per_quantile.keys():

            all_SP = []

            for model in epsilon_rashomon_per_quantile[q]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                PPR = eval.compute_pred_pos_ratio(CM)
                SP = eval.compute_Statistical_Parity(PPR, model_part=model_part)
                all_SP.append(SP['over_all_groups'])


            mean_SP.append(float(np.mean(all_SP)))
            std_SP.append(float(np.std(all_SP)))
        
        x_pos = np.arange(len(epsilon_rashomon_per_quantile.keys()))

        plt.plot(
            x_pos,
            mean_SP,
            label = f'{method}',
            linewidth=2, color = color_map[method]
        )

        plt.fill_between(
            x_pos,
            np.array(mean_SP) - np.array(std_SP),
            np.array(mean_SP) + np.array(std_SP),
            color='gray',
            alpha=0.25
        )

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

    plt.xticks(x_pos, labels)
    plt.legend()
    plt.xlabel(f'Quantiles of Coverage Rate')
    plt.ylabel('Max. Statistical Parity')
    plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'SP'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_SP{Dataset_name}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def individual_arbitrariness_all_methods(Dataset_name,seed, result_dir,epsilon,split,n_quantiles, bins):
    """This function generates a plot comparing the average individual arbitrariness 
    across different methods for a given dataset. 
    It computes the Rashomon sets for each method, 
    calculates the average individual arbitrariness for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the average individual arbitrariness across
      quantiles for each method on the same graph.

    """

    methods = [
        'HybridCORELSPreClassifier',
        'HybridCORELSPostClassifier',
        'HyRS',
        'CRL'
    ]


    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    color_map = {
        'HybridCORELSPreClassifier': 'tab:blue',
        'HybridCORELSPostClassifier': 'tab:orange',
        'HyRS': 'tab:green',
        'CRL': 'tab:red'
    }

    for m_idx, method in enumerate(methods):

        # --- compute Rashomon sets ---

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # --- compute arbitrariness ---
        for i, q in enumerate(epsilon_rashomon_per_quantile.keys()):
            ax = axes[i]

            pred_types_list = []
            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1) #average of predcition type across Rashomon set
            
            # ax.hist(
            #     arbitrariness,
            #     bins=20,
            #     alpha=0.4,
            #     color=color_map[method],
            #     label=method
            # )
            ax.hist(
                arbitrariness,
                bins=20,
                histtype='step',
                linewidth=2,
                color=color_map[method],
                label=method
                #density=True   
            )


    # -------------------------------
    # Formatting per subplot
    # -------------------------------
    for i, ax in enumerate(axes):
        lower = quantiles[i]
        upper = quantiles[i+1]

        label = f"[{lower:.2f}, {upper:.2f})" if i < 3 else f"[{lower:.2f}, {upper:.2f}]"

        #ax.set_title(f"Coverage: {label}")
        ax.set_xlabel("Individual Arbitrariness")
        #ax.set_ylabel("Frequency")
        ax.set_ylabel("# Data Points")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.text(
        0.07, 0.95,                      # position (relative to axes)
        f"Coverage: {label}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top'
        )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    #fig.suptitle(f"{dataset_name.capitalize()} | Individual Arbitrariness", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Individual_arbitrariness_{Dataset_name}.png"
    plt.savefig(output_file, bbox_inches="tight")

    #plt.show()
    plt.close()


def CDF_individual_arbitrariness_all_methods(Dataset_name,seed, result_dir,epsilon,split,n_quantiles, bins):
    """This function generates a plot comparing the average individual arbitrariness 
    across different methods for a given dataset. 
    It computes the Rashomon sets for each method, 
    calculates the average individual arbitrariness for models in each quantile of coverage rate, and plot it via a CDF
    .

    """

    methods = [
        'HybridCORELSPreClassifier',
        'HybridCORELSPostClassifier',
        'HyRS',
        'CRL'
    ]


    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    color_map = {
        'HybridCORELSPreClassifier': 'tab:blue',
        'HybridCORELSPostClassifier': 'tab:orange',
        'HyRS': 'tab:green',
        'CRL': 'tab:red'
    }

    for m_idx, method in enumerate(methods):

        # --- compute Rashomon sets ---

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # --- compute arbitrariness ---
        for i, q in enumerate(epsilon_rashomon_per_quantile.keys()):
            ax = axes[i]

            pred_types_list = []
            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1) #average of predcition type across Rashomon set
            # Sort values for empirical CDF
            x = np.sort(arbitrariness)
            y = np.arange(1, len(x) + 1) / len(x)

            ax.step(
                x,
                y,
                where='post',
                linewidth=2,
                color=color_map[method],
                label=method
            )
                
        

    # -------------------------------
    # Formatting per subplot
    # -------------------------------
    for i, ax in enumerate(axes):
        lower = quantiles[i]
        upper = quantiles[i+1]

        label = f"[{lower:.2f}, {upper:.2f})" if i < 3 else f"[{lower:.2f}, {upper:.2f}]"
        ax.set_xlabel("Individual Arbitrariness")
        ax.set_ylabel("Proportion of Data Points")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Coverage: {label}", fontsize=10)
        ax.grid(True)
        # ax.text(
        # 0.07, 0.95,                      # position (relative to axes)
        # f"Coverage: {label}",
        # transform=ax.transAxes,
        # fontsize=10,
        # verticalalignment='top'
        # )

    # Shared legend
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='lower center', ncol=4)

    #fig.suptitle(f"{dataset_name.capitalize()} | Individual Arbitrariness", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"CDF_Individual_arbitrariness_{Dataset_name}.png"
    plt.savefig(output_file, bbox_inches="tight")

    #plt.show()
    plt.close()

def CDF_individual_arbitrariness_all_methods_separate(
    Dataset_name, seed, result_dir, epsilon, split, n_quantiles, bins
):
    """Generate one CDF plot per coverage quantile for individual arbitrariness."""

    methods = [
        'HybridCORELSPreClassifier',
        'HybridCORELSPostClassifier',
        'HyRS',
        'CRL'
    ]

    color_map = {
        'HybridCORELSPreClassifier': 'tab:blue',
        'HybridCORELSPostClassifier': 'tab:orange',
        'HyRS': 'tab:green',
        'CRL': 'tab:red'
    }

    output_dir = Path.cwd() / 'plots' / 'Arbit'
    output_dir.mkdir(exist_ok=True)

    # Store CDF data for each quantile and method
    cdf_data = {q_idx: {} for q_idx in range(n_quantiles)}
    quantiles = None

    for method in methods:

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
            Dataset_name,
            method,
            result_dir,
            seed=seed,
            n_quantiles=n_quantiles,
            bins=bins,
            epsilon=epsilon
        )

        for q_idx, q in enumerate(epsilon_rashomon_per_quantile.keys()):

            pred_types_list = []

            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1)

            x = np.sort(arbitrariness)
            y = np.arange(1, len(x) + 1) / len(x)

            cdf_data[q_idx][method] = (x, y)

    # Create and save one plot per quantile
    for q_idx in range(n_quantiles):

        fig, ax = plt.subplots(figsize=(6, 4))

        for method in methods:
            if method not in cdf_data[q_idx]:
                continue

            x, y = cdf_data[q_idx][method]

            ax.step(
                x,
                y,
                where='post',
                linewidth=2,
                color=color_map[method],
                label=method
            )

        lower = quantiles[q_idx]
        upper = quantiles[q_idx + 1]

        coverage_label = (
            f"[{lower:.2f}, {upper:.2f})"
            if q_idx < n_quantiles - 1
            else f"[{lower:.2f}, {upper:.2f}]"
        )

        ax.set_title(f"Coverage: {coverage_label}", fontsize=13)
        ax.set_xlabel("Individual Arbitrariness", fontsize = 14)
        ax.set_ylabel("Proportion of Data Points",fontsize = 14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True)
        #ax.legend(loc="best", fontsize=9)

        plt.tight_layout()

        output_file = (
            output_dir
            / f"CDF_Individual_arbitrariness_{Dataset_name}_q{q_idx + 1}.png"
        )

        #plt.show()
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()



def ICA_box_all_methods (Dataset_name,seed, result_dir,epsilon,split,n_quantiles, bins):
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange',
    'HyRS': 'tab:green',
    'CRL': 'tab:red'
    }

    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24
    for i,method in enumerate(ESTIMATORS):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        

        all_ICA = []

        for q in epsilon_rashomon_per_quantile.keys():

            pred_types_list = []

            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1)
            ICA = 1 - 2*np.abs((arbitrariness-0.5))
            all_ICA.append(ICA)


        num_methods = len(ESTIMATORS)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods


        # --- BOX PLOTS ---
        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_ICA,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[method])
            patch.set_alpha(0.7)



    #plt.legend()

    plt.xlabel(f'Transparency bins', fontsize= LABEL_SIZE)
    plt.ylabel('ICA', fontsize= LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
        # --- LEGENDS ---
    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(ESTIMATORS)
    # ]

    #plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"ICA_Box_{Dataset_name}_{split}_{epsilon}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    plt.show()
    plt.close()  



def sparity_all_methods(Dataset_name,seed, result_dir, epsilon, n_quantiles, bins):
    """This function generates a plot comparing the sparsity (number of rules) 
    across different methods for a given dataset. 
    It computes the Rashomon sets for each method, 
    calculates the number of rules for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the number of rules across
      quantiles for each method on the same graph.

    """
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed
    )
    color_map = {
        'HybridCORELSPreClassifier': 'tab:blue',
        'HybridCORELSPostClassifier': 'tab:orange',
        'HyRS': 'tab:green',
        'CRL': 'tab:red'
    }

    for method in ESTIMATORS:

        #Compute Rashomon sets
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        
        # -------------------------------
        # Compute statistics
        # -------------------------------


        sparsity_mean = []
        sparsity_std = []
        for q in epsilon_rashomon_per_quantile.keys():
            sparsity_all = []
            for model in epsilon_rashomon_per_quantile[q]:
                if method!='HyRS':
                    sparsity_all.append(len(model['rules'])) #for now only for HybridCOREL
                else:
                    pos,neg = model['rules']
                    sparsity_all.append(len(pos)+len(neg))

            sparsity_mean.append(np.mean(sparsity_all))
            sparsity_std.append(np.std(sparsity_all))


        x_pos = np.arange(len(epsilon_rashomon_per_quantile.keys()))

        plt.plot(
            x_pos,
            sparsity_mean,
            linewidth=2, label = f'{method}', color = color_map[method]
        )

        plt.fill_between(
            x_pos,
            np.array(sparsity_mean) - np.array(sparsity_std),
            np.array(sparsity_mean) + np.array(sparsity_std),
            color='gray',
            alpha=0.25
        )


    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

    plt.xticks(x_pos, labels)
    plt.legend()

    plt.xlabel(f'Quantiles of Coverage Rate')
    plt.ylabel('Number of Rules')
    #plt.title(f'{Dataset_name.capitalize()}')

    plt.grid()
    plt.tight_layout()
    #--- save ---
    output_dir = Path.cwd()/'plots'/'Sparsity'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Sparsity_{Dataset_name}.png"
    plt.savefig(output_file, bbox_inches="tight")

    #plt.show()
    plt.close()

################################
#Statistical tests
################################
def statistcial_ICF(Dataset_name,method,seed, result_dir, epsilon,split, demographic_group, n_quantiles, bins, print_summary=False, actual_cov = True):
    """this fucntion assumes that the ICF over all actual coverages have a qudratic nonlinear structure and test this hypothesis"""
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)

    # -------------------------------
    # Collect Rashomon models
    # -------------------------------

    epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
    Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
    )
    # -------------------------------
    # Compute statistics
    # -------------------------------
    
    conditions = my_data.demographicGroup(summarized=True)[demographic_group]

    eval = Evaluation(X[split], features, conditions)


    all_icf_disparity_per_quantile = []
    actual_coverage = [] 

    for q in epsilon_rashomon_per_quantile.keys():

        all_icf_disparity = []
        
        for model in epsilon_rashomon_per_quantile[q]:
            actual_coverage.append(model[f'coverage_rate_{split}'])
            ICF = eval.compute_fairness(model[f'preds_types_{split}'])
            ICF_disparity = eval.compute_ICF_disparity(ICF)

            all_icf_disparity.append(ICF_disparity['over_all_groups'])

        all_icf_disparity_per_quantile.append(all_icf_disparity)
    all_quntiles_index = [[i+1]* len(j) for i,j in enumerate(all_icf_disparity_per_quantile)]
    flattend_quantile_index = np.array([j for i in all_quntiles_index for j in i])
    flattend_ICF = np.array([j for i in all_icf_disparity_per_quantile for j in i])
    assert len(flattend_quantile_index) == len(flattend_ICF)
    assert len(actual_coverage) == len(flattend_ICF)
    #to fit a quadratic model to see if there is nonlinear relationship between quantiles and ICF disparity
    if actual_cov: 
        x = np.array(actual_coverage)
    else:
        x = flattend_quantile_index
    y = flattend_ICF
    df = pd.DataFrame({"x": x, "x2": x**2, "y": y})
    X = sm.add_constant(df[["x", "x2"]])
    model = sm.OLS(df["y"], X).fit()
    
    if print_summary:
        print(model.summary())
    peak = -model.params["x"] / (2 * model.params["x2"])
    
    beta2 = model.params["x2"]
    pval2 = model.pvalues["x2"]

    if (pval2 < 0.05) and (beta2 < 0):
        peak = -model.params["x"] / (2 * beta2)
        valid_peak = (1 <= peak <= n_quantiles)
        #print("quadratic term is significant and negative")
        state = 'Bell_shaped'
        return state, beta2, pval2, peak #if valid_peak else None
    else:
        #fit a linear model if quadratic term is not significant or if the peak is not within the range of quantiles
        X_lin = sm.add_constant(x)
        model_lin = sm.OLS(y, X_lin).fit()

        beta1 = model_lin.params[1]
        pval1 = model_lin.pvalues[1]
        if pval1 < 0.05 and beta1 > 0:
            state = 'positive_linear'
            #print("there is a significant positive linear relationship")
        if pval1 < 0.05 and beta1 < 0:
            #print("there is a significant negative linear relationship")
            state = 'negative_linear'
        if pval1 >= 0.05:
            state = 'no_relation'
        return state, beta1, pval1, None

def all_icf_info_per_quantile (Dataset_name, method, seed,result_dir, epsilon,split, n_quantiles, bins, demographic_group):
    """ this fucntion gives all Max ICF disparity values for all quantiles as all_icf_disparity_per_quantile"""
    seed = 0 #seed is always zero
    np.random.seed(seed)
    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)

    # -------------------------------
    # Collect Rashomon models
    # -------------------------------

    epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
    Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
    )
    # -------------------------------
    # Compute statistics
    # -------------------------------

    conditions = my_data.demographicGroup(summarized=True)[demographic_group]

    eval = Evaluation(X[split], features, conditions)


    all_icf_disparity_per_quantile = []
    actual_coverage = [] 

    for q in sorted(epsilon_rashomon_per_quantile.keys()):

        all_icf_disparity = []
        
        for model in epsilon_rashomon_per_quantile[q]:
            actual_coverage.append(model[f'coverage_rate_{split}'])
            ICF = eval.compute_fairness(model[f'preds_types_{split}'])
            ICF_disparity = eval.compute_ICF_disparity(ICF)

            all_icf_disparity.append(ICF_disparity['over_all_groups'])

        all_icf_disparity_per_quantile.append(all_icf_disparity)
    all_quntiles_index = [[i+1]* len(j) for i,j in enumerate(all_icf_disparity_per_quantile)]
    flattend_quantile_index = np.array([j for i in all_quntiles_index for j in i])
    flattend_ICF = np.array([j for i in all_icf_disparity_per_quantile for j in i])
    assert len(flattend_quantile_index) == len(flattend_ICF)
    assert len(actual_coverage) == len(flattend_ICF)
    #to fit a quadratic model to see if there is nonlinear relationship between quantiles and ICF disparity
    return all_icf_disparity_per_quantile


def pairwise_adjacent_tests(icf_by_quantile, alternative="two-sided"):
    """
    icf_by_quantile: list of arrays, [Q1, Q2, ..., QK]
    alternative: "two-sided", "greater", or "less"
    returns list of dicts with stats per adjacent pair
    """
    results = []
    for i in range(len(icf_by_quantile) - 1):
        q1 = np.asarray(icf_by_quantile[i])
        q2 = np.asarray(icf_by_quantile[i+1])

        stat, p = mannwhitneyu(q1, q2, alternative=alternative)

        # Direction via medians (robust)
        med1, med2 = np.median(q1), np.median(q2)
        if med2 > med1:
            direction = "increase"
        elif med2 < med1:
            direction = "decrease"
        else:
            direction = "no_change"

        results.append({
            "pair": f"Q{i+1}→Q{i+2}",
            "U": stat,
            "p_value": p,
            "median_Qi": med1,
            "median_Qi+1": med2,
            "direction": direction,
            "n1": len(q1),   
            "n2": len(q2)
        })
    return results


def adjust_pvalues(results, alpha=0.05, method="holm"):
    pvals = [r["p_value"] for r in results]
    reject, pvals_adj, _, _ = multipletests(pvals, alpha=alpha, method=method)
    for r, pa, rej in zip(results, pvals_adj, reject):
        r["p_adj"] = pa
        r["significant"] = bool(rej)
    return results


def add_effect_size(results):
    for r in results:
        U = r["U"]
        n1 = r.get("n1")
        n2 = r.get("n2")
        # if you store lengths:
        # n1, n2 = len(q1), len(q2)
        mean_U = n1*n2/2
        std_U = np.sqrt(n1*n2*(n1+n2+1)/12)
        z = (U - mean_U) / std_U
        r["effect_r"] = z / np.sqrt(n1 + n2)
    return results

def summarize_pattern(results):
    seq = []
    for r in results:
        if (np.abs(r["effect_r"])>=0.2 ) and r["significant"]:
            seq.append( r["direction"]+"_B") 
        elif r["significant"] :
            seq.append( r["direction"]+"_NB") 
        else:
            seq.append("No_change")  # no significant change
    return "-".join(seq)

################################
#Analysis over Rashomon sets of Quantiles AFTER Mitigations
################################
LABEL_SIZE = 18
TICK_SIZE = 15

def Fair_max_delta_ICF_all_methods (Dataset_name,seed, epsilon,split, demographic_group, n_quantiles, bins=None):
    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                    'HybridCORELSPost':'HybridCORELSPostClassifier'}
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
    }
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24
  
 

    for i,j in enumerate(method_match.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
    
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)


        all_icf_disparity_per_quantile = []
        

        for q in epsilon_rashomon_per_quantile.keys():
            all_icf_disparity = []
            

            for model in epsilon_rashomon_per_quantile[q]:
                ICF = eval.compute_fairness(model[f'preds_types_{split}'])
                ICF_disparity = eval.compute_ICF_disparity(ICF)

                all_icf_disparity.append(ICF_disparity['over_all_groups'])

            all_icf_disparity_per_quantile.append(all_icf_disparity)



        num_methods = len(method_match)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods


        # --- BOX PLOTS ---
        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_icf_disparity_per_quantile,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[j])


    #fontsize=LABEL_SIZE
    plt.xlabel(f'Transparency bins', fontsize=LABEL_SIZE)
    plt.ylabel('ICD', fontsize=LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")


    # --- LEGENDS ---
    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(method_match)
    # ]

    # plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize=TICK_SIZE) #, fontsize=TICK_SIZE
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_Max_Delta_ICF{Dataset_name}_{demographic_group}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def save_fairness_methods_legend():

    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',        # light blue
        'HybridCORELSPost': '#ffbb78'        # light orange
    }

    label_map = {
        'HybridCORELSPre_Fair': 'HybridCORELSPre (with ICD mitigation)',
        'HybridCORELSPost_Fair': 'HybridCORELSPost (with ICD mitigation)',
        'HybridCORELSPre': 'HybridCORELSPre',
        'HybridCORELSPost': 'HybridCORELSPost'
    }

    legend_elements = [
        Line2D(
            [0], [0],
            color=color_map[method],
            lw=3,
            label=label_map[method]
        )
        for method in color_map
    ]

    # Create standalone legend figure
    legend_fig = plt.figure(figsize=(8, 0.8))

    legend_fig.legend(
        handles=legend_elements,
        loc='center',
        ncol=4,
        frameon=False,
        fontsize=14
    )

    output_dir = Path.cwd() / 'plots' / 'legends'
    output_dir.mkdir(parents=True, exist_ok=True)

    legend_fig.savefig(
        output_dir / 'HybridCORELS_fairness_shared_legend.pdf',
        bbox_inches='tight',
        dpi=300
    )

    plt.close(legend_fig)


def Fair_max_EO_all_methods (Dataset_name,seed, epsilon,split, demographic_group, model_part, n_quantiles, bins=None):
   
    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                    'HybridCORELSPost':'HybridCORELSPostClassifier'}

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
    }
    
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24

    for i,j in enumerate(method_match.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
        
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)

        all_EO_per_quantile = []
        for q in epsilon_rashomon_per_quantile.keys():

            all_EO = []

            for model in epsilon_rashomon_per_quantile[q]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                TPR = eval.compute_true_pos_ratio(CM)
                EO = eval.compute_Equal_Opportunity(TPR, model_part=model_part)
                all_EO.append(EO['over_all_groups'])
            all_EO_per_quantile.append(all_EO)

        num_methods = len(method_match)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods

        # dynamic colors
        #colors = plt.cm.tab10(np.linspace(0, 1, num_methods))


        # --- BOX PLOTS (left axis) ---


        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_EO_per_quantile,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[j])



    plt.xlabel(f'Transparency bins', fontsize = LABEL_SIZE)
    plt.ylabel('Equal Opportunity', fontsize = LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")


    # # --- LEGENDS ---
    # from matplotlib.patches import Patch


    # # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(method_match)
    # ]

    # plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'EO'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_Max_EO{Dataset_name}_{demographic_group}_{split}_{epsilon}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Fair_max_SP_all_methods (Dataset_name,seed, epsilon,split, demographic_group, model_part, n_quantiles, bins=None):
   
    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                    'HybridCORELSPost':'HybridCORELSPostClassifier'}

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
    }
    
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24

    for i,j in enumerate(method_match.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
        
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        
        conditions = my_data.demographicGroup(summarized=True)[demographic_group]

        eval = Evaluation(X[split], features, conditions)

        all_SP_per_quantile = []
        for q in epsilon_rashomon_per_quantile.keys():

            all_SP = []

            for model in epsilon_rashomon_per_quantile[q]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                PPR = eval.compute_pred_pos_ratio(CM)
                SP = eval.compute_Statistical_Parity(PPR, model_part=model_part)
                all_SP.append(SP['over_all_groups'])
            all_SP_per_quantile.append(all_SP)

        num_methods = len(method_match)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods

        # dynamic colors
        #colors = plt.cm.tab10(np.linspace(0, 1, num_methods))


        # --- BOX PLOTS (left axis) ---


        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_SP_per_quantile,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[j])



    plt.xlabel(f'Transparency bins', fontsize = LABEL_SIZE)
    plt.ylabel('Statistical Parity' , fontsize = LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")


    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(method_match)
    # ]

    # plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'SP'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_Max_SP{Dataset_name}_{demographic_group}_{split}_{epsilon}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Fair_Sparsity_all_methods (Dataset_name,seed, epsilon, demographic_group, n_quantiles, bins=None):

    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                    'HybridCORELSPost':'HybridCORELSPostClassifier'}

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
    }
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24
    for i,j in enumerate(method_match.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # -------------------------------
        # Compute statistics
        # -------------------------------

        all_SP_per_quantile = []
        for q in epsilon_rashomon_per_quantile.keys():

            all_Sparsity = []

            for model in epsilon_rashomon_per_quantile[q]:
                if method_match[j]!='HyRS': #in case , I want to use it for HyRS and CRL in future
                    all_Sparsity.append(len(model['rules'])) #for now only for HybridCOREL
                else:
                    pos,neg = model['rules']
                    all_Sparsity.append(len(pos)+len(neg))
                
            all_SP_per_quantile.append(all_Sparsity)

        num_methods = len(method_match)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods

        # dynamic colors
        #colors = plt.cm.tab10(np.linspace(0, 1, num_methods))


        # --- BOX PLOTS (left axis) ---


        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_SP_per_quantile,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[j])



    plt.xlabel(f'Transparency bins', fontsize= LABEL_SIZE)
    plt.ylabel('Number of Rules', fontsize=LABEL_SIZE )
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")


    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(method_match)
    # ]

    # plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize= TICK_SIZE)
    plt.yticks(fontsize= TICK_SIZE)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Sparsity'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_Sparsity{Dataset_name}_{demographic_group}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Fair_individual_arbitrariness_all_methods(Dataset_name,seed,epsilon,split,demographic_group,n_quantiles, bins):

    """This function generates a plot comparing the average individual arbitrariness 
    across different methods for a given dataset. 
    It computes the Rashomon sets for each method, 
    calculates the average individual arbitrariness for models in each quantile of coverage rate, 
    and plots the mean and standard deviation of the average individual arbitrariness across
      quantiles for each method on the same graph.

    """



    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                'HybridCORELSPre':'HybridCORELSPreClassifier',
                'HybridCORELSPost':'HybridCORELSPostClassifier'}
   
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for meth_indx, j in enumerate(list(method_match.keys())[:2]):

        # --- compute Rashomon sets ---

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # --- compute arbitrariness ---
        for i, q in enumerate(epsilon_rashomon_per_quantile.keys()):
            ax = axes[i]

            pred_types_list = []
            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1) #average of predcition type across Rashomon set
            
            # ax.hist(
            #     arbitrariness,
            #     bins=20,
            #     alpha=0.4,
            #     color=color_map[method],
            #     label=method
            # )
            ax.hist(
                arbitrariness,
                bins=20,
                histtype='step',
                linewidth=2,
                color=color_map[j],
                label=method
                #density=True   
            )


    # -------------------------------
    # Formatting per subplot
    # -------------------------------
    for i, ax in enumerate(axes):
        lower = quantiles[i]
        upper = quantiles[i+1]

        label = f"[{lower:.2f}, {upper:.2f})" if i < 3 else f"[{lower:.2f}, {upper:.2f}]"

        #ax.set_title(f"Coverage: {label}")
        ax.set_xlabel("Individual Arbitrariness")
        #ax.set_ylabel("Frequency")
        ax.set_ylabel("# Data Points")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.text(
        0.07, 0.95,                      # position (relative to axes)
        f"Coverage: {label}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top'
        )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    #fig.suptitle(f"{dataset_name.capitalize()} | Individual Arbitrariness", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_Individual_arbitrariness_{Dataset_name}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")

    #plt.show()
    plt.close()


def CDF_Fair_individual_arbitrariness_all_methods(Dataset_name,seed,epsilon,split,demographic_group,n_quantiles, bins):
   
    """This function generates a plot comparing the average individual arbitrariness 
    across different methods for a given dataset. 


    """

    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                'HybridCORELSPre':'HybridCORELSPreClassifier',
                'HybridCORELSPost':'HybridCORELSPostClassifier'}

    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    uncertain_results = {method: [] for method in method_match}
    for meth_indx, j in enumerate(list(method_match.keys())):

        # --- compute Rashomon sets ---

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # --- compute arbitrariness ---
        
        for i, q in enumerate(epsilon_rashomon_per_quantile.keys()):
            if not len(epsilon_rashomon_per_quantile[q]):
                continue
            ax = axes[i]

            pred_types_list = []
            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1) #average of predcition type across Rashomon set
            uncertain_mass = np.mean((arbitrariness > 0.3) & (arbitrariness < 0.7))
            uncertain_results[j].append(uncertain_mass)
            x = np.sort(arbitrariness)
            y = np.arange(1, len(x) + 1) / len(x)

            ax.step(
                x,
                y,
                where='post',
                linewidth=2,
                color=color_map[j],
                label=j
            )
    #this is to reveal uncertain_results information
    # for method in method_match:
    #     print(f"\n{method}")
    #     for i, val in enumerate(uncertain_results[method]):
    #         print(f"  Q{i+1}: {val:.3f}")
    # -------------------------------
    # Formatting per subplot
    # -------------------------------
    for i, ax in enumerate(axes):
        lower = quantiles[i]
        upper = quantiles[i+1]

        label = f"[{lower:.2f}, {upper:.2f})" if i < 3 else f"[{lower:.2f}, {upper:.2f}]"
        ax.set_xlabel("Individual Arbitrariness")
        ax.set_ylabel("Proportion of Data Points")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.text(
        0.07, 0.95,                      # position (relative to axes)
        f"Coverage: {label}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top'
        )

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2)

    #fig.suptitle(f"{dataset_name.capitalize()} | Individual Arbitrariness", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"CDF_Fair_Individual_arbitrariness_{Dataset_name}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def CDF_Fair_individual_arbitrariness_withmass(Dataset_name,seed,epsilon,split,demographic_group,n_quantiles, bins):


    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                'HybridCORELSPre':'HybridCORELSPreClassifier',
                'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                'HybridCORELSPost':'HybridCORELSPostClassifier'}

    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
        }
    method_label = {
        'HybridCORELSPre_Fair': 'HybridCORELSPre (with ICD mitigation)',
        'HybridCORELSPost_Fair': 'HybridCORELSPost (with ICD mitigation)',
        'HybridCORELSPre': 'HybridCORELSPre',
        'HybridCORELSPost': 'HybridCORELSPre'
        }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    uncertain_results = {method: [] for method in method_match}
    for meth_indx, j in enumerate(list(method_match.keys())):

        # --- compute Rashomon sets ---

        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
        # --- compute arbitrariness ---
        
        for i, q in enumerate(epsilon_rashomon_per_quantile.keys()):
            if not len(epsilon_rashomon_per_quantile[q]):
                continue
            ax = axes[i]

            pred_types_list = []
            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1) #average of predcition type across Rashomon set
            uncertain_mass = np.mean((arbitrariness > 0.3) & (arbitrariness < 0.7))
            uncertain_results[j].append(uncertain_mass)
            x = np.sort(arbitrariness)
            y = np.arange(1, len(x) + 1) / len(x)

            ax.step(
                x,
                y,
                where='post',
                linewidth=2,
                color=color_map[j],
                label=f"{method_label[j]}: {100 * uncertain_mass:.1f}%"
            )
    #this is to reveal uncertain_results information
    # for method in method_match:
    #     print(f"\n{method}")
    #     for i, val in enumerate(uncertain_results[method]):
    #         print(f"  Q{i+1}: {val:.3f}")
    # -------------------------------
    # Formatting per subplot
    # -------------------------------
    for i, ax in enumerate(axes):
        lower = quantiles[i]
        upper = quantiles[i+1]

        label = f"[{lower:.2f}, {upper:.2f})" if i < 3 else f"[{lower:.2f}, {upper:.2f}]"
        ax.set_title(f"Coverage: {label}", fontsize=12)
        ax.set_xlabel("Individual Arbitrariness", fontsize = 18)
        ax.set_ylabel("Proportion of Data Points", fontsize = 18)
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ax.text(
        # 0.07, 0.95,                      # position (relative to axes)
        # f"Coverage: {label}",
        # transform=ax.transAxes,
        # fontsize=10,
        # verticalalignment='top'
        # )
        #uncomment it you want the legened
        # ax.legend(
        #     loc='best', #"lower right"
        #     fontsize=10,
        #     frameon=True,
        #     title="Uncertain (0.3, 0.7)",
        #     title_fontsize=8
        # )

    # # Shared legend
    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=2)

    #fig.suptitle(f"{dataset_name.capitalize()} | Individual Arbitrariness", fontsize=14)

    plt.tight_layout()

    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"CDF_Fair_Individual_arbitrariness_mass_{Dataset_name}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Predictive_Performance_all_methods (Dataset_name,seed, epsilon,split, demographic_group, n_quantiles, bins=None):

    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                    'HybridCORELSPost':'HybridCORELSPostClassifier'}

    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
    }
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24

    for i,j in enumerate(method_match.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------

        all_train_acc_per_quantile = []
        for q in epsilon_rashomon_per_quantile.keys():

            all_train_acc = []

            for model in epsilon_rashomon_per_quantile[q]:
                all_train_acc.append(model[f'acc_{split}'])

            all_train_acc_per_quantile.append(all_train_acc)

        num_methods = len(method_match)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods

        # dynamic colors
        #colors = plt.cm.tab10(np.linspace(0, 1, num_methods))


        # --- BOX PLOTS (left axis) ---


        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_train_acc_per_quantile,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[j])



    plt.xlabel(f'Transparency bins', fontsize = LABEL_SIZE)
    plt.ylabel('Accuracy', fontsize = LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")


    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(method_match)
    # ]

    # plt.legend(handles=legend_elements, loc='lower left')
    plt.xticks(base_positions, labels, fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Accuracy'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_PredictivePerformance_{Dataset_name}_{demographic_group}_{split}_{epsilon}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Fair_ICA_all_methods (Dataset_name,seed, epsilon,split,demographic_group, n_quantiles, bins=None):
    
    result_dir ={'HybridCORELSPre_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPost_Fair': Path.cwd()/"Mitigation_results"/demographic_group,
                'HybridCORELSPre': Path.cwd()/"bootstrap_results",
                'HybridCORELSPost': Path.cwd()/"bootstrap_results"}
    method_match = {'HybridCORELSPre_Fair':'HybridCORELSPreClassifier',
                    'HybridCORELSPre':'HybridCORELSPreClassifier',
                    'HybridCORELSPost_Fair':'HybridCORELSPostClassifier',
                    'HybridCORELSPost':'HybridCORELSPostClassifier'}
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)
    color_map = {
        'HybridCORELSPre_Fair': '#1f77b4',   # strong blue
        'HybridCORELSPost_Fair': '#ff7f0e',  # strong orange
        'HybridCORELSPre': '#aec7e8',             # light blue
        'HybridCORELSPost': '#ffbb78'             # light orange
    }
    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 24
    TICK_SIZE = 20
    LEGEND_SIZE = 18
    TITLE_SIZE = 24
  
 

    for i,j in enumerate(method_match.keys()):
        # -------------------------------
        # Collect Rashomon models
        # -------------------------------
    
        epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon(
        Dataset_name, method_match[j], result_dir[j], seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon
        )
            # -------------------------------
        # Compute statistics
        # -------------------------------
        

        all_ICA = []

        for q in epsilon_rashomon_per_quantile.keys():

            pred_types_list = []

            for model in epsilon_rashomon_per_quantile[q]:
                pred_types_list.append(model[f'preds_types_{split}'])

            if len(pred_types_list) == 0:
                continue

            all_models_pred_type = np.column_stack(pred_types_list)
            arbitrariness = np.mean(all_models_pred_type, axis=1)
            ICA = 1 - 2*np.abs((arbitrariness-0.5))
            all_ICA.append(ICA)

        num_methods = len(method_match)
        num_quantiles = len(epsilon_rashomon_per_quantile.keys())

        base_positions = np.arange(1, num_quantiles + 1)

        # dynamic spacing
        total_width = 0.8
        width = total_width / num_methods


        # --- BOX PLOTS ---
        # center all groups
        positions = base_positions + (i - (num_methods - 1)/2) * width

        box = plt.boxplot(
            all_ICA,
            positions=positions,
            widths=width * 0.9,
            showmeans=True,
            whis=[0, 100],
            patch_artist=True
        )

        for patch in box['boxes']:
            patch.set_facecolor(color_map[j])


    #fontsize=LABEL_SIZE
    plt.xlabel(f'Transparency bins', fontsize=LABEL_SIZE)
    plt.ylabel('ICA', fontsize=LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')

    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")


    # --- LEGENDS ---
    # boxplot legend (groups)
    # legend_elements = [
    #     Patch(facecolor=color_map[method], label=method)
    #     for i, method in enumerate(method_match)
    # ]

    # plt.legend(handles=legend_elements, loc='upper left')
    plt.xticks(base_positions, labels, fontsize=TICK_SIZE) #, fontsize=TICK_SIZE
    plt.yticks(fontsize=TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Arbit'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Fair_ICA{Dataset_name}_{demographic_group}_{split}_{epsilon}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


################################
#Analysis over mitigated methods VS. max_coverage
################################
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from pathlib import Path

def save_ICD_mitigation_legend():

    color_map = {
        'HybridCORELSPre (with ICD mitigation)': 'tab:blue',
        'HybridCORELSPost (with ICD mitigation)': 'tab:orange'
    }

    legend_elements = [
        Line2D(
            [0], [0],
            color=color,
            lw=3,
            label=label
        )
        for label, color in color_map.items()
    ]

    legend_fig = plt.figure(figsize=(6, 0.8))

    legend_fig.legend(
        handles=legend_elements,
        loc='center',
        ncol=2,
        frameon=False,
        fontsize=16
    )

    output_dir = Path.cwd() / 'plots' / 'legends'
    output_dir.mkdir(parents=True, exist_ok=True)

    legend_fig.savefig(
        output_dir / 'ICD_mitigation_shared_legend.pdf',
        bbox_inches='tight',
        dpi=300
    )

    plt.close(legend_fig)

def Acc_Max_Coverage_one_quantile (Dataset_name, epsilon,split,seed, demographic_group, n_quantiles,all_max_cov, quantile ,bins=None):
    
    mean = []
    std = []
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange'}
    label_fair = {'HybridCORELSPostClassifier': 'HybridCORELSPost_Fair', 
                'HybridCORELSPreClassifier' : 'HybridCORELSPre_Fair'}

    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 28
    TICK_SIZE = 24
    LEGEND_SIZE = 18
    TITLE_SIZE = 24

    for method in ['HybridCORELSPostClassifier', 'HybridCORELSPreClassifier']:
        mean = []
        std = []
        for max_cov in all_max_cov:
            if max_cov == 0.05:
                result_dir = Path.cwd()/'Mitigation_results'/'Gender'
            elif max_cov == 1:
                result_dir = Path.cwd()/'bootstrap_results'
            else: 
                result_dir = Path.cwd()/'Mitigation_results'/'MGender'
            
            epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon, max_cov= None if max_cov in [0.05, 1] else max_cov)
            
            mean.append (np.mean([model[f'acc_{split}'] for model in epsilon_rashomon_per_quantile[quantile]]))
            std.append( np.std([model[f'acc_{split}'] for model in epsilon_rashomon_per_quantile[quantile]]))
  

        plt.plot(
            all_max_cov,
            mean,
            label = label_fair[method],
            linewidth=2, color = color_map[method],
            marker = '*',
        )

        plt.fill_between(
            all_max_cov,
            np.array(mean) - np.array(std),
            np.array(mean) + np.array(std),
            color= color_map[method],
            alpha=0.25
        )
    n_q = n_quantiles
    i = list(epsilon_rashomon_per_quantile.keys()).index(quantile)
    label = (
            f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
    # Global title
    plt.title(f"Coverage: {label}", fontsize=16)


    #plt.legend()
    plt.xlabel(f'Max. ICD Constraint $\eta$' , fontsize = LABEL_SIZE)
    plt.ylabel('Accuracy', fontsize = LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')
    plt.xscale('log')
    #plt.xticks(all_max_cov, [f"{x:g}" for x in all_max_cov], ) #rotation=45, fontsize = TICK_SIZE
    plt.xticks(fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Fair'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Acc_maxcoverage_{Dataset_name}_{demographic_group}_{quantile if n_quantiles==4 else 'oneq'}_{split}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()

                
def Sparsity_Max_Coverage_one_quantile (Dataset_name, epsilon,seed, demographic_group, n_quantiles,all_max_cov, quantile ,bins=None):

    mean = []
    std = []
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange'}
    label_fair = {'HybridCORELSPostClassifier': 'HybridCORELSPost_Fair', 
                'HybridCORELSPreClassifier' : 'HybridCORELSPre_Fair'}

    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 28
    TICK_SIZE = 24
    for method in ['HybridCORELSPostClassifier', 'HybridCORELSPreClassifier']:
        mean = []
        std = []
        for max_cov in all_max_cov:
            if max_cov == 0.05:
                result_dir = Path.cwd()/'Mitigation_results'/'Gender'
            elif max_cov == 1:
                result_dir = Path.cwd()/'bootstrap_results'
            else: 
                result_dir = Path.cwd()/'Mitigation_results'/'MGender'
            
            epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon, max_cov= None if max_cov in [0.05, 1] else max_cov)
            
            mean.append (np.mean([len(model['rules']) for model in epsilon_rashomon_per_quantile[quantile]]))
            std.append( np.std([len(model['rules']) for model in epsilon_rashomon_per_quantile[quantile]]))
  

        plt.plot(
            all_max_cov,
            mean,
            label = label_fair[method],
            linewidth=2, color = color_map[method],
            marker = '*',
        )

        plt.fill_between(
            all_max_cov,
            np.array(mean) - np.array(std),
            np.array(mean) + np.array(std),
            color= color_map[method],
            alpha=0.25
        )
    n_q = n_quantiles
    i = list(epsilon_rashomon_per_quantile.keys()).index(quantile)
    label = (
            f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
    # Global title
    plt.title(f"Coverage: {label}", fontsize=16)


    #plt.legend()
    plt.xlabel(f'Max. ICD Constraint $\eta$', fontsize = LABEL_SIZE)
    plt.ylabel('Number of Rules', fontsize = LABEL_SIZE)
    plt.ylim(bottom=0)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')
    plt.xscale('log')
    # plt.xticks(all_max_cov, [f"{x:g}" for x in all_max_cov])
    plt.xticks(fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Fair'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Sparsity_maxcoverage_{Dataset_name}_{demographic_group}_{quantile if n_quantiles==4 else 'oneq'}_{split}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    # plt.show()
    plt.close()

                
def Max_DeltaICF_Max_Coverage_one_quantile(Dataset_name,seed, epsilon, split, demographic_group, n_quantiles,all_max_cov, quantile ,bins=None):
  
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)

    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange'}
    label_fair = {'HybridCORELSPostClassifier': 'HybridCORELSPost_Fair', 
                'HybridCORELSPreClassifier' : 'HybridCORELSPre_Fair'}

    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 28
    TICK_SIZE = 24
    for method in ['HybridCORELSPostClassifier', 'HybridCORELSPreClassifier']:
        mean = []
        std = []
        for max_cov in all_max_cov:
            if max_cov == 0.05:
                result_dir = Path.cwd()/'Mitigation_results'/'Gender'
            elif max_cov == 1:
                result_dir = Path.cwd()/'bootstrap_results'
            else: 
                result_dir = Path.cwd()/'Mitigation_results'/'MGender'
            
            epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon, max_cov= None if max_cov in [0.05, 1] else max_cov)
            
            n_q = len(epsilon_rashomon_per_quantile)
            q = quantile 

            conditions = my_data.demographicGroup(summarized=True)[demographic_group]

            eval = Evaluation(X[split], features, conditions)

            all_icf_disparity = []

            for model in epsilon_rashomon_per_quantile[q]:
                ICF = eval.compute_fairness(model[f'preds_types_{split}'])
                ICF_disparity = eval.compute_ICF_disparity(ICF)

                all_icf_disparity.append(ICF_disparity['over_all_groups'])


            mean.append(np.mean(all_icf_disparity))
            std.append(np.std(all_icf_disparity))



        plt.plot(
            all_max_cov,
            mean,
            label = label_fair[method],
            linewidth=2, color = color_map[method],
            marker = '*',
        )

        plt.fill_between(
            all_max_cov,
            np.array(mean) - np.array(std),
            np.array(mean) + np.array(std),
            color= color_map[method],
            alpha=0.25)
        
    n_q = n_quantiles
    i = list(epsilon_rashomon_per_quantile.keys()).index(quantile)
    label = (
            f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
    # Global title
    plt.title(f"Coverage: {label}", fontsize=16)


    #plt.legend()
    plt.xlabel(f'Max. ICD Constraint $\eta$', fontsize = LABEL_SIZE)
    plt.ylabel('ICD', fontsize = LABEL_SIZE)
    plt.ylim(bottom=0)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')
    plt.xscale('log')
    # plt.xticks(all_max_cov, [f"{x:g}" for x in all_max_cov])
    plt.xticks(fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Fair'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_Delta_ICF_MaxCoevrage_{Dataset_name}_{demographic_group}_{quantile if n_quantiles==4 else 'oneq'}_{split}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Max_EO_Max_Coverage_one_quantile(Dataset_name,seed, epsilon, split, demographic_group, n_quantiles,all_max_cov, quantile ,bins=None):
  
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)


    mean = []
    std = []
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange'}
    label_fair = {'HybridCORELSPostClassifier': 'HybridCORELSPost_Fair', 
                'HybridCORELSPreClassifier' : 'HybridCORELSPre_Fair'}

    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 28
    TICK_SIZE = 24
    for method in ['HybridCORELSPostClassifier', 'HybridCORELSPreClassifier']:
        mean = []
        std = []
        for max_cov in all_max_cov:
            if max_cov == 0.05:
                result_dir = Path.cwd()/'Mitigation_results'/'Gender'
            elif max_cov == 1:
                result_dir = Path.cwd()/'bootstrap_results'
            else: 
                result_dir = Path.cwd()/'Mitigation_results'/'MGender'
            
            epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon, max_cov= None if max_cov in [0.05, 1] else max_cov)
            
            n_q = len(epsilon_rashomon_per_quantile)
            q = quantile 

            conditions = my_data.demographicGroup(summarized=True)[demographic_group]

            eval = Evaluation(X[split], features, conditions)

            all_EO = []

            for model in epsilon_rashomon_per_quantile[q]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                TPR = eval.compute_true_pos_ratio(CM)
                EO = eval.compute_Equal_Opportunity(TPR, model_part='TPR_overal')
                all_EO.append(EO['over_all_groups'])

            mean.append(np.mean(all_EO))
            std.append(np.std(all_EO))

        plt.plot(
            all_max_cov,
            mean,
            label = label_fair[method],
            linewidth=2, color = color_map[method],
            marker = '*',
        )

        plt.fill_between(
            all_max_cov,
            np.array(mean) - np.array(std),
            np.array(mean) + np.array(std),
            color= color_map[method],
            alpha=0.25)
        
    n_q = n_quantiles
    i = list(epsilon_rashomon_per_quantile.keys()).index(quantile)
    label = (
            f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
    # Global title
    plt.title(f"Coverage: {label}", fontsize=16)


    #plt.legend()
    plt.xlabel(f'Max. ICD Constraint $\eta$', fontsize= LABEL_SIZE)
    plt.ylabel('Equal Opportunity',  fontsize= LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')
    plt.xscale('log')
    # plt.xticks(all_max_cov, [f"{x:g}" for x in all_max_cov])
    plt.xticks(fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.ylim(bottom=0)
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Fair'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_EO_MaxCoverage_{Dataset_name}_{demographic_group}_{quantile if n_quantiles==4 else 'oneq'}_{split}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


def Max_SP_Max_Coverage_one_quantile(Dataset_name,seed, epsilon, split, demographic_group, n_quantiles,all_max_cov, quantile ,bins=None):
  
    seed = 0 
    np.random.seed(seed)

    train_proportion = 0.8

    my_data = Dataset.from_csv(
        Path.cwd().parent / f'examples/data/{Dataset_name}_mined.csv',
        Dataset_name
    )
    my_data.pre_process()

    X, y, features, prediction = my_data.get_data_norulemining(
        {"train": train_proportion, "test": 1-train_proportion},
        random_state_param=seed)


    mean = []
    std = []
    color_map = {
    'HybridCORELSPreClassifier': 'tab:blue',
    'HybridCORELSPostClassifier': 'tab:orange'}
    label_fair = {'HybridCORELSPostClassifier': 'HybridCORELSPost_Fair', 
                'HybridCORELSPreClassifier' : 'HybridCORELSPre_Fair'}

    plt.figure(figsize=(10, 6))
    LABEL_SIZE = 28
    TICK_SIZE = 24
    for method in ['HybridCORELSPostClassifier', 'HybridCORELSPreClassifier']:
        mean = []
        std = []
        for max_cov in all_max_cov:
            if max_cov == 0.05:
                result_dir = Path.cwd()/'Mitigation_results'/'Gender'
            elif max_cov == 1:
                result_dir = Path.cwd()/'bootstrap_results'
            else: 
                result_dir = Path.cwd()/'Mitigation_results'/'MGender'
            
            epsilon_rashomon_per_quantile, _, quantiles = generate_quantiles_Rashomon (Dataset_name, method, result_dir, seed=seed, n_quantiles=n_quantiles, bins=bins, epsilon=epsilon, max_cov= None if max_cov in [0.05, 1] else max_cov)
            
            n_q = len(epsilon_rashomon_per_quantile)
            q = quantile 

            conditions = my_data.demographicGroup(summarized=True)[demographic_group]

            eval = Evaluation(X[split], features, conditions)

            all_SP = []

            for model in epsilon_rashomon_per_quantile[q]:
                CM = eval.confusion_matrix(model[f'preds_{split}'],y[split],model[f'preds_types_{split}'])
                PPR = eval.compute_pred_pos_ratio(CM)
                SP = eval.compute_Statistical_Parity(PPR, model_part='PPR_overal')
                all_SP.append(SP['over_all_groups'])

            mean.append(np.mean(all_SP))
            std.append(np.std(all_SP))

        plt.plot(
            all_max_cov,
            mean,
            label = label_fair[method],
            linewidth=2, color = color_map[method],
            marker = '*',
        )

        plt.fill_between(
            all_max_cov,
            np.array(mean) - np.array(std),
            np.array(mean) + np.array(std),
            color= color_map[method],
            alpha=0.25)
        
    n_q = n_quantiles
    i = list(epsilon_rashomon_per_quantile.keys()).index(quantile)
    label = (
            f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})"
            if i < n_q - 1
            else f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")
    # Global title
    plt.title(f"Coverage: {label}", fontsize=16)


    #plt.legend()
    plt.xlabel(f'Max. ICD Constraint $\eta$', fontsize= LABEL_SIZE)
    plt.ylabel('Statistical Parity',  fontsize= LABEL_SIZE)
    #plt.title(f'{Dataset_name.capitalize()} | {demographic_group}')
    plt.xscale('log')
    # plt.xticks(all_max_cov, [f"{x:g}" for x in all_max_cov])
    plt.xticks(fontsize = TICK_SIZE)
    plt.yticks(fontsize = TICK_SIZE)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.ylim(bottom=0)
    # --- save ---
    output_dir = Path.cwd()/'plots'/'Fair'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Max_SP_MaxCoverage_{Dataset_name}_{demographic_group}_{quantile if n_quantiles==4 else 'oneq'}_{split}.pdf"
    plt.savefig(output_file, bbox_inches="tight")
    #plt.show()
    plt.close()


if __name__ == '__main__':

    result_dir = Path.cwd()/"bootstrap_results"
    DATASETS = ["compas", "adult", "acs_employ"]
    Dataset_name = 'compas'
    groups = ['Gender', 'Age', 'Race']
    method = 'HybridCORELSPostClassifier'
    epsilon = 0.01
    split = 'test'
    seed = 0
    




    #################################
    #Analysis over quantiles
    #################################
   
    # for dataset in DATASETS:
    #     ICA_box_all_methods (Dataset_name = dataset,seed= seed, result_dir=result_dir,epsilon=epsilon,split=split,n_quantiles=4, bins=None)
        #CDF_individual_arbitrariness_all_methods_separate (Dataset_name=dataset,seed=seed, result_dir=result_dir,epsilon=epsilon,split=split,n_quantiles=4, bins=None)
        # CDF_individual_arbitrariness_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir,epsilon=epsilon,split=split,n_quantiles=4, bins=None)    
        #sparity_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, n_quantiles=4, bins=None)
        # for group in groups:
            #max_delta_ICF_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=0.20, split=split, demographic_group=group, n_quantiles=4, bins=None)
            #max_delat_ICF_all_methods_withfairs(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, split=split, demographic_group=group, n_quantiles=4, bins=None)
            # max_delta_ICF_all_methods_box (Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon= epsilon, split=split, demographic_group=group, n_quantiles=4, bins=None)
            # max_EO_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, split=split, demographic_group=group, model_part='TPR_overal', n_quantiles=4, bins=None)
            # max_SP_all_methods (Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, split=split, demographic_group=group, model_part='PPR_overal', n_quantiles=4, bins=None)
            # for method in ESTIMATORS:
            #     comprehensive_ICF_plot_2_axis_quantiles(Dataset_name=dataset, method=method, seed=seed, result_dir=result_dir, n_quantiles=4 ,epsilon=epsilon,bins=None,split=split, demographic_group=group)
    #save_methods_legend()

    ################################
    #Statistical Analysis over quantiles
    ################################
    # method = 'CRL' 
    # #HybridCORELSPostClassifier
    # for dataset in DATASETS:
    #     for group in ['Age', 'Gender', 'Race']:
    #         all_icf_disparity_per_quantile = all_icf_info_per_quantile (dataset, method, seed,result_dir, epsilon=epsilon,split =split, n_quantiles=4, bins=None, demographic_group= group)
    #         results = pairwise_adjacent_tests (all_icf_disparity_per_quantile)
    #         results = adjust_pvalues(results)
    #         add_effect_size(results)
    #         print(dataset,method,group, summarize_pattern(results))


    #################################
    #Analysis After Mitigation
    #################################

    # for dataset in DATASETS:
    #     for group in groups:
            # Fair_max_delta_ICF_all_methods (Dataset_name = dataset,seed = seed, epsilon = epsilon,split = split, demographic_group=group, n_quantiles=4, bins=None)
            # Fair_max_EO_all_methods (Dataset_name = dataset,seed = seed, epsilon = epsilon,split = split, demographic_group= group, model_part= 'TPR_overal', n_quantiles=4, bins=None)
            # Fair_max_SP_all_methods (Dataset_name = dataset,seed = seed, epsilon = epsilon,split = split, demographic_group= group, model_part= 'PPR_overal', n_quantiles=4, bins=None)
            # Fair_Sparsity_all_methods (Dataset_name = dataset,seed = seed, epsilon = epsilon, demographic_group = group, n_quantiles = 4, bins = None)
    #       CDF_Fair_individual_arbitrariness_all_methods(Dataset_name = dataset,seed = seed,epsilon = epsilon,split = split,demographic_group= group,n_quantiles=4, bins=None)
            #CDF_Fair_individual_arbitrariness_withmass(Dataset_name = dataset,seed = seed,epsilon = epsilon,split = split,demographic_group = group,n_quantiles=4, bins=None) #only for Gender
            # Fair_ICA_all_methods (Dataset_name = dataset,seed = seed,epsilon = epsilon,split = split,demographic_group = group,n_quantiles=4, bins=None)
            # Predictive_Performance_all_methods (Dataset_name = dataset,seed = seed, epsilon = epsilon,split = split, demographic_group=group, n_quantiles=4, bins=None)
    #save_fairness_methods_legend()

    



   #################################
    #Analysis for Mitigated methods VS. max coverage constraint
    #################################
    #this analysis is supposed to be done only for gender and one datasets for now but the code is general
    Dataset_name = 'compas'
    demographic_group = 'Gender'
    all_max_cov = [0.01, 0.03, 0.05,0.07, 0.10,0.12, 0.15, 0.25,1]
    for q in ['q1' , 'q2', 'q3', 'q4']: #,,'q2', 'q3', 'q4'
        Acc_Max_Coverage_one_quantile (Dataset_name, epsilon=epsilon,split=split,seed=seed, demographic_group=demographic_group,\
                                        n_quantiles=4,all_max_cov= all_max_cov, quantile=q,bins=None)

        Sparsity_Max_Coverage_one_quantile(Dataset_name=Dataset_name,seed=seed, epsilon=epsilon,\
                                    demographic_group=demographic_group, n_quantiles=4,all_max_cov=all_max_cov, quantile=q ,bins=None)
        Max_DeltaICF_Max_Coverage_one_quantile(Dataset_name=Dataset_name,seed=seed, epsilon=epsilon,split = split,\
                                     demographic_group=demographic_group, n_quantiles=4,all_max_cov=all_max_cov, quantile=q ,bins=None)
        Max_EO_Max_Coverage_one_quantile(Dataset_name=Dataset_name,seed=seed, epsilon=epsilon,split = split,\
                                    demographic_group=demographic_group, n_quantiles=4,all_max_cov=all_max_cov, quantile=q ,bins=None)
        Max_SP_Max_Coverage_one_quantile(Dataset_name=Dataset_name,seed=seed, epsilon=epsilon,split = split,\
                                    demographic_group=demographic_group, n_quantiles=4,all_max_cov=all_max_cov, quantile=q ,bins=None)
        
    #save_ICD_mitigation_legend()