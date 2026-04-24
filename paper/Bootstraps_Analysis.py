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

    ax1.set_ylabel('Interpretability Coverage')
    ax1.set_xlabel(f'Quantiles of Coverage Rate')

    # --- SECOND AXIS (right) ---
    ax2 = ax1.twinx()

    ax2.plot(
        base_positions,
        mean_icf_disparity,
        color='black',
        marker='o',
        linewidth=2,
        label='Max. IC Disparity'
    )

    ax2.fill_between(
        base_positions,
        np.array(mean_icf_disparity) - np.array(std_icf_disparity),
        np.array(mean_icf_disparity) + np.array(std_icf_disparity),
        color='gray',
        alpha=0.2
    )

    ax2.set_ylabel('Max. Interpretability Coverage Disparity')

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
    plt.title(f'{Dataset_name.capitalize()} | {method}')
    ax1.set_xticks(base_positions)
    
    #to generate quantiles range labels
    labels = []
    for i in range(len(quantiles) - 1):
        if i < len(quantiles) - 2:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f})")
        else:
            labels.append(f"[{quantiles[i]:.2f}, {quantiles[i+1]:.2f}]")

    ax1.set_xticklabels(labels)

    ax1.grid(axis='y')
    plt.tight_layout()
    # # --- save ---
    output_dir = Path.cwd()/'plots'/'ICF'
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"Comp_ICF_2axis{Dataset_name}_{method}_{demographic_group}.png"
    plt.savefig(output_file, bbox_inches="tight")
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
#Analysis over Rashomon sets of Quantiles AFTER Mitigations
################################

#for Race 






if __name__ == '__main__':

    result_dir = Path.cwd()/"bootstrap_results"
    DATASETS = ["compas", "adult", "acs_employ"]
    groups = ['Gender', 'Age', 'Race']
    method = 'HybridCORELSPostClassifier'
    epsilon = 0.01 
    split = 'train'
    seed = 0
    #comprehensive_ICF_plot(dataset_name='compas', method='HybridCORELSPreClassifier',epsilon=0.01, 
                        #    demographic_group='Age',split='train')
    # comprehensive_ICF_plot_two_axis(dataset_name='compas', method='CRL',epsilon=0.01, 
    #                        demographic_group='Age',split='train')
    #max_delta_ICF (dataset_name='compas', method='HybridCORELSPreClassifier',epsilon=0.01, split='train') 
    # delta_ICF_for_pairs (dataset_name='compas', method='HybridCORELSPreClassifier',epsilon=0.01, split='train',
    #                      demographic_group='Age') 
    #max_EO (dataset_name='compas', method='HybridCORELSPostClassifier',epsilon=0.01, split='train', model_part='TPR_overal')
    # EO_for_pairs (dataset_name='compas', method='CRL',epsilon=0.01, split='train',
    #                        demographic_group='Age', model_part='TPR_overal')
    #max_SP(dataset_name='compas', method='HybridCORELSPreClassifier',epsilon=0.01, split='train', model_part='PPR_overal')

    # SP_for_pairs (dataset_name='compas', method='HybridCORELSPreClassifier',epsilon=0.01, split='train',
    #                       demographic_group='Age', model_part='PPR_overal')
    
    #individual_arbitrariness(dataset_name,method, epsilon,split='train')
    #group_arbitrariness (dataset_name,method,epsilon,split='train',demographic_group='Race')

    #sparsity (dataset_name,method,epsilon)


    #################################
    #Analysis over quantiles
    #################################
    
    for dataset in DATASETS:
        # individual_arbitrariness_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir,epsilon=epsilon,split=split,n_quantiles=4, bins=None)    
        #sparity_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, n_quantiles=4, bins=None)
        # for group in groups:
        #     max_delta_ICF_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, split=split, demographic_group=group, n_quantiles=4, bins=None)
            # max_EO_all_methods(Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, split=split, demographic_group=group, model_part='TPR_overal', n_quantiles=4, bins=None)
            # max_SP_all_methods (Dataset_name=dataset,seed=seed, result_dir=result_dir, epsilon=epsilon, split=split, demographic_group=group, model_part='PPR_overal', n_quantiles=4, bins=None)
            # for method in ESTIMATORS:
            #     comprehensive_ICF_plot_2_axis_quantiles(Dataset_name=dataset, method=method, seed=seed, result_dir=result_dir, n_quantiles=4 ,epsilon=0.01,bins=None,split='train', demographic_group=group)

