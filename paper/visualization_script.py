
from pathlib import Path
from exp_utils import read_json, Dataset, paired_subgroups, save_json
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.stats import linregress , spearmanr
import pandas as pd
import numpy as np
import math



################################
#All Utils 

################################

def load_results(dataset, method):
    """This fucntion reads all experiments per seed for one dataset and one method

    Args:
        dataset (str): name of the dataset
        method (str): name of the method

    Returns:
        list: list of all results 
    """
    results = []

    for f in result_dir.glob(f"{dataset}__{method}__*.json"):
        r = read_json(f)
        results.append(r)

    return results


def get_TPR_one_data (data, split):
    """This function gets one experimnet per seed and returns TPR 

    Args:
        data (_type_): _description_
        split (_type_): _description_

    Returns:
        dict: returns TPR for all conditions
    """
    conds = list(data['trans_total'][split].keys())
    TPR = {cond:{'overall':[],'T':[],'B':[]} for cond in conds}
    for i in conds:
        if isinstance(data['trans_total'][split][i]['TP']['T'], list):
            TP_T = np.array(data['trans_total'][split][i]['TP']['T'])
            TP_B = np.array(data['trans_total'][split][i]['TP']['B'])
            FN_T = np.array(data['trans_total'][split][i]['FN']['T'])
            FN_B = np.array(data['trans_total'][split][i]['FN']['B'])
            TPR[i]['overall'] = (np.divide((TP_T+TP_B), ((TP_T+TP_B)+(FN_T+FN_B)), out=np.zeros_like(TP_T, dtype=float), where=((TP_T+TP_B)+(FN_T+FN_B))!=0)).tolist()
            TPR[i]['T'] = (np.divide(TP_T, (TP_T+FN_T), out=np.zeros_like(TP_T, dtype=float), where=(TP_T+FN_T)!=0)).tolist()
            TPR[i]['B'] =( np.divide(TP_B, (TP_B+FN_B), out=np.zeros_like(TP_B, dtype=float), where=(TP_B+FN_B)!=0)).tolist()
    return TPR


def make_split_dict(conds):
    return {
        'acc': [],
        'ICF': {c: [] for c in conds},
        'TPR_O': {c: [] for c in conds},
        'TPR_T': {c: [] for c in conds},
        'TPR_B': {c: [] for c in conds},
        'cov': [],
        'param': [], 
        'pos_ratio_T': {c: [] for c in conds},
        'pos_ratio_B': {c: [] for c in conds},
    }


def aggregate_results (results):
    by_seed = {seed: {'train': make_split_dict(conds), 'test': make_split_dict(conds)} for seed in range(n_seeds)}

    for r in results:
        seed = r["seed"]
        TPR = {split: get_TPR_one_data(r, split) for split in ['train', 'test']}

        for split in ['train', 'test']:
            if  isinstance(r['accuracy'][split], list):
                by_seed[seed][split]['cov'].extend(r['coverage'][split])
                by_seed[seed][split]['acc'].extend(r['accuracy'][split])
            else:
                by_seed[seed][split]['cov'].append(r['coverage'][split])
                by_seed[seed][split]['acc'].append(r['accuracy'][split])
            
            for tpr_type, key in zip(['overall', 'T', 'B'], ['TPR_O', 'TPR_T', 'TPR_B']):
                for cond in conds:
                    by_seed[seed][split][key][cond].extend(TPR[split][cond][tpr_type])
            
            for cond in conds:
                by_seed[seed][split]['ICF'][cond].extend(r['trans_total'][split][cond]['ICF'])
            
            for pos_type, key in zip(['T','B'], ['pos_ratio_T','pos_ratio_B']):
                for cond in conds:
                    by_seed[seed][split][key][cond].extend(r['trans_total'][split][cond]['Pos_Ratio'][pos_type])

            by_seed[seed][split]['param'].append(r[TRADEOFF_PARAM[r['model']]])
    
    return by_seed




def interpolate_metric(x, y, kind='linear',agg='mean', fill_value='extrapolate'):
    """
    Interpolate y over x (automatically sorts x).
    
    Parameters:
        x: array-like, coverage values
        y: array-like, metric values
        kind: interpolation type ('linear', 'cubic', etc.)
        fill_value: how to handle extrapolation
    
    Returns:
        f: interp1d function
    """
    x = np.array(x)
    y = np.array(y)
    
    # sort by x
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    y_sorted = y[sorted_idx]

   

    # Group by coverage as there might be several values for each coverage (more than one accuarcy for one coverage)
    bucket = defaultdict(list)
    for c, v in zip(x_sorted, y_sorted):
        bucket[c].append(v)

    unique_x = np.array(sorted(bucket.keys()))

    if agg == 'mean':
        agg_values = np.array([np.mean(bucket[c]) for c in unique_x])
    elif agg == 'max':
        agg_values = np.array([np.max(bucket[c]) for c in unique_x])
    else:
        raise ValueError("agg must be 'mean' or 'max'")


    if len(unique_x) < 2:
            return lambda x: np.full_like(x, agg_values[0], dtype=float)

    return interp1d(
        unique_x,
        agg_values,
        kind=kind,
        bounds_error=False,
        fill_value=fill_value
    )




def clean_nans(obj):
    if isinstance(obj, list):
        return [clean_nans(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, float) and math.isnan(obj):
        return 0
    return obj
    
################################

n_seeds = 10

TRADEOFF_VALUES = {
    "HybridCORELSPreClassifier": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "HybridCORELSPostClassifier": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    "CRL": [
        0.001, 0.0016681, 0.00278256, 0.00464159, 0.00774264,
        0.0129155, 0.02154435, 0.03593814, 0.05994843, 0.1
    ],
    "HyRS": [
        0.001, 0.00215443, 0.00464159, 0.01, 0.02154435,
        0.04641589, 0.1, 0.21544347, 0.46415888, 1.0
    ],
}


TRADEOFF_PARAM = {
    "HybridCORELSPreClassifier": "min_coverage",
    "HybridCORELSPostClassifier": "min_coverage",
    "CRL": "alpha",
    "HyRS": "beta",
}


experiments = {'data': ["compas", "adult", "acs_employ"], 
               'method':["HybridCORELSPreClassifier","HybridCORELSPostClassifier", "CRL","HyRS"]
               , 'tardeoff_value':TRADEOFF_VALUES.values(),'seed': list(range(n_seeds))}


#should be changed for all datasets and all methods :
dataset_name = 'adult'
method = 'CRL'


# Load data
my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
my_data.pre_process()

conditions = my_data.demographicGroup()
conds = conditions['All']


#read all results for one dataset and one method 
result_dir = Path.cwd()/'results_1'

results = load_results(dataset_name, method)

results = clean_nans(results)


#group by all data per seed
by_seed = aggregate_results(results)


# Store interpolated functions per seed and split
interpolated_by_seed = {k: {'train': {}, 'test': {}} for k in range(n_seeds)}


for seed in by_seed:
    for split in ['train', 'test']:
        cov = by_seed[seed][split]['cov']
        
        # Accuracy
        acc = by_seed[seed][split]['acc']
        interpolated_by_seed[seed][split]['acc'] = interpolate_metric(cov, acc) #fill_value= np.nan
        
        # ICF per condition
        interpolated_by_seed[seed][split]['ICF'] = {}
        for cond in conds:
            icf = by_seed[seed][split]['ICF'][cond]
            interpolated_by_seed[seed][split]['ICF'][cond] = interpolate_metric(cov, icf)
        
        # TPR metrics per condition
        for tpr_type in ['TPR_O', 'TPR_T', 'TPR_B']:
            interpolated_by_seed[seed][split][tpr_type] = {}
            for cond in conds:
                tpr = by_seed[seed][split][tpr_type][cond]
                interpolated_by_seed[seed][split][tpr_type][cond] = interpolate_metric(cov, tpr)
        
        for pos_type, key in zip(['T','B'], ['pos_ratio_T','pos_ratio_B']):
            interpolated_by_seed[seed][split][key] = {}
            for cond in conds:
                pos_ratio = by_seed[seed][split][key][cond]
                interpolated_by_seed[seed][split][key][cond] = interpolate_metric(cov, pos_ratio)
        
        # Coverage itself (just in case)
        interpolated_by_seed[seed][split]['cov'] = interpolate_metric(cov, cov)


min_cov = max(np.min(by_seed[s][split]['cov']) for s in range(n_seeds))
max_cov = min(np.max(by_seed[s][split]['cov']) for s in range(n_seeds))
desired_cov = np.linspace(min_cov, max_cov, 50)

#Interpolation is restricted to transparency regions where subgroup-level ICF is defined,
# ensuring stable estimation of ΔICF and its relationship with outcome disparities.
#desired_cov = np.linspace(0.1, 0.975, 50)  # 50 points from 0 to 1

out_dir = Path("Plots")
out_dir.mkdir(exist_ok=True)

#############################################

#Accuracy Vs. Coverage

#############################################

# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# for i, split in enumerate(['train', 'test']):

#     ax = axes[i]
    
#     acc = np.array([interpolated_by_seed[s][split]['acc'](desired_cov) for s in range(n_seeds)])
#     mean_acc = np.mean(acc, axis=0)
#     std_acc = np.std(acc, axis=0)
    
#     ax.plot(desired_cov, mean_acc)
#     ax.fill_between(desired_cov, mean_acc - std_acc, mean_acc + std_acc, alpha=0.25)
    
    
#     ax.set_xlabel("Transparency")
#     ax.set_ylabel("Accuracy")
#     ax.set_title(f'{split.capitalize()}')
   
#     ax.legend()
#     ax.grid(True)

# out_dir = Path("Plots")
# out_dir.mkdir(exist_ok=True)
# plt.tight_layout()
# file_path = out_dir / f"Accuracy_transparency_{method}_{dataset_name}.png"
# plt.savefig(file_path)
# plt.show()
# plt.close()


# #############################################

# #ICF for all subgroups in each demographic groups

# #############################################


# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     sub_info = {'freq': my_data.get_condition_freq(conditions[disparity]),
#              'corr': my_data.corr_subgroups(conditions[disparity])}

#     for i, split in enumerate(['train', 'test']):
#         ax = axes[i]
#         for cond in conditions[disparity]:
#             ICF = np.array([interpolated_by_seed[s][split]['ICF'][cond](desired_cov) for s in range(n_seeds)])
#             mean = np.mean(ICF, axis=0)
#             std = np.std(ICF, axis=0)

#             ax.plot(desired_cov, mean, label = f"{cond} frequency={sub_info['freq'][cond]:.2f}, correlation={sub_info['corr'][cond]:.1f}")
#             ax.fill_between(desired_cov, mean - std, mean + std, alpha=0.25)


#             ax.set_xlabel("Transparency")
#             ax.set_ylabel("Interpretability Coverage Fairness")
#             ax.set_title(f'{split.capitalize()}')

#             ax.legend()
#             ax.grid(True)
#     plt.tight_layout()
#     file_path = out_dir / f"ICF_transparency_{disparity}_{method}_{dataset_name}.png"
#     plt.savefig(file_path)
#     plt.show()
#     plt.close()

##########################################

#Global Fiarness : Signed TPR Difference for each pair of demographic groups

#########################################

# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:
#     for i,j in paired_subgroups(conditions[disparity]):
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         for s,split in enumerate(['train', 'test']):
#             ax = axes[s]

#             delta_tpr_all = {k:[] for k in ['TPR_O','TPR_B','TPR_T']}
#             for s in range(n_seeds):
#                 for tpr_type in ['TPR_O','TPR_B','TPR_T']:
#                     delta = interpolated_by_seed[s][split][tpr_type][i](desired_cov)-interpolated_by_seed[s][split][tpr_type][j](desired_cov) 
#                     delta_tpr_all[tpr_type].append( delta) 
#             delta_tpr_all = {k:np.array(delta_tpr_all[k]) for k in ['TPR_O','TPR_B','TPR_T'] }
#             mean = {k:np.mean(delta_tpr_all[k], axis=0) for k in ['TPR_O','TPR_B','TPR_T']}
#             std = {k:np.std(delta_tpr_all[k], axis=0) for k in ['TPR_O','TPR_B','TPR_T']}


#             styles = {'TPR_O': dict(lw=2.5, ls='-'),
#                       'TPR_T': dict(lw=1.5, ls='--', alpha=0.8),
#                       'TPR_B': dict(lw=1.5, ls=':', alpha=0.8),}
#             for tpr_type in ['TPR_O','TPR_B','TPR_T']:
#                 ax.plot(desired_cov, mean[tpr_type],
#                          label = {'TPR_O': 'Overall (Hybrid)','TPR_T': 'Transparent Part','TPR_B': 'Black-box Part'}[tpr_type],
#                          **styles[tpr_type]) 
#                 if tpr_type == 'TPR_O':
#                     ax.fill_between(desired_cov, mean[tpr_type] - std[tpr_type], mean[tpr_type] + std[tpr_type], alpha=0.25)


#                 ax.set_xlabel("Transparency")
#                 ax.set_ylabel(f"Signed TPR Difference {i}-{j}")
#                 ax.set_title(f"{split.capitalize()} — Positive ⇒ {i} favored")

#                 ax.legend()
#                 ax.grid(True)
#                 ax.axhline(0, color='black', lw=1, alpha=0.6)
#         plt.tight_layout()
#         file_path = out_dir / f"Signed_Disparity_{i}-{j}_{method}_{dataset_name}.png"
#         plt.savefig(file_path)
#         #plt.show()
#         plt.close()



# ##########################################

# #Global Fiarness : Maximum TPR Gap 

# #########################################



# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     for ii, split in enumerate(['train', 'test']):
#         ax = axes[ii]
#         all_delta= []
#         for s in range(n_seeds):
#             delta_seed = []
#             for i,j in paired_subgroups(conditions[disparity]):
#                 gap = np.abs(interpolated_by_seed[s][split]['TPR_O'][i](desired_cov)-interpolated_by_seed[s][split]['TPR_O'][j](desired_cov))
#                 delta_seed.append(gap)
#             delta_seed = np.max(delta_seed, axis= 0)
#             all_delta.append(delta_seed)
#         all_delta = np.array(all_delta)
#         mean = np.mean(all_delta, axis = 0)
#         std = np.std(all_delta, axis = 0)

#         ax.plot(desired_cov, mean, label = f"Maximum TPR difference")
#         ax.fill_between(desired_cov, mean - std, mean + std, alpha=0.25)


#         ax.set_xlabel("Transparency")
#         ax.set_ylabel(f"Maximum TPR difference for {disparity.capitalize()}")
#         ax.set_title(f'{split.capitalize()}')

#         ax.legend()
#         ax.grid(True)

#     plt.tight_layout()
#     file_path = out_dir / f"Max_Disparity of_{disparity}_{method}_{dataset_name}.png"
#     plt.savefig(file_path)
#     #plt.show()
#     plt.close()


# ##########################################

# #Global Fiarness :TPR Vs. ICF for each subgroup relative to a reference 
#TPR_g - TPR_reference VS ICF_g

# #########################################




# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:

#     freq = my_data.get_condition_freq(conditions[disparity])
#     reference = conditions[disparity][np.argmax(freq.values())]
#     for cond in conditions[disparity]:
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         for i, split in enumerate(['train', 'test']):
#             ax = axes[i]
#             ICF = np.array([interpolated_by_seed[s][split]['ICF'][cond](desired_cov) for s in range(n_seeds)])
#             mean = np.mean(ICF, axis=0)
#             std = np.std(ICF, axis=0)
#             #delta TPR
#             delta_tpr = np.array([interpolated_by_seed[s][split]['TPR_O'][cond](desired_cov)-interpolated_by_seed[s][split]['TPR_O'][reference](desired_cov)\
#                     for s in range(n_seeds)])
#             mean_delta = np.mean(delta_tpr, axis=0)
#             std_delta = np.std(delta_tpr, axis=0)
#             sc = ax.scatter(mean,mean_delta,c=desired_cov,cmap="viridis",s=40,alpha=0.8)
#             cbar = plt.colorbar(sc, ax=ax)
#             cbar.set_label("Transparency (coverage)")
#             ax.set_xlabel(f"Interpretability Coverage Fairness (ICF) ({cond})")
#             ax.set_ylabel(f"TPR Difference ({cond} - {reference})")
#             ax.set_title(f'{split.capitalize()}')

#             #to fit a trend line and show Pearson correlation 
#             slope, intercept, r, p, _ = linregress(mean, mean_delta)
#             x_line = np.linspace(mean.min(), mean.max(), 100)
#             y_line = slope * x_line + intercept
#             ax.plot(x_line,y_line,linestyle="--",color="black",linewidth=2,label=f"Linear fit (r={r:.2f})")
#             ax.legend()

#             #to show uncertainty by faint error bar (with computed std)
#             ax.errorbar(mean,mean_delta,xerr=std,yerr=std_delta,fmt="none",ecolor="gray",alpha=0.3,capsize=0)
#             ax.axhline(0, color="red", linestyle=":", linewidth=1)



#         plt.tight_layout()
#         file_path = out_dir / f"ICF_{cond}_vs_TPR_{method}_{dataset_name}.png"
#         plt.savefig(file_path)
#         #plt.show()
#         plt.close()





##########################################
# Global Fairness: Maximum TPR Gap
# One plot per dataset × demographic
# 4 methods together, Train/Test subplots
##########################################

# datasets = ["compas", "adult", "acs_employ"]
# methods = [
#     "HybridCORELSPreClassifier",
#     "HybridCORELSPostClassifier",
#     "CRL",
#     "HyRS"
# ]

# demographic_groups = ["Age", "Gender", "Race"]

# out_dir = Path("Plots/MaxTPRGap")
# out_dir.mkdir(parents=True, exist_ok=True)

# for dataset_name in datasets:

#     # Load dataset once
#     my_data = Dataset.from_csv(
#         Path.cwd().parent / f"examples/data/{dataset_name}_mined.csv",
#         dataset_name
#     )
#     my_data.pre_process()

#     conditions = my_data.demographicGroup()
#     conds = conditions["All"]

#     for disparity in demographic_groups:

#         fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

#         for method in methods:

#             # ---- Load & aggregate results (your existing pipeline) ----
#             result_dir = Path.cwd() / "results"
#             results = load_results(dataset_name, method)
#             if len(results) == 0:
#                 continue  # skip missing experiments

#             by_seed = aggregate_results(results)

#             interpolated_by_seed = {s: {"train": {}, "test": {}} for s in range(n_seeds)}

#             for seed in by_seed:
#                 for split in ["train", "test"]:
#                     cov = by_seed[seed][split]["cov"]

#                     interpolated_by_seed[seed][split]["TPR_O"] = {}
#                     for cond in conds:
#                         tpr = by_seed[seed][split]["TPR_O"][cond]
#                         interpolated_by_seed[seed][split]["TPR_O"][cond] = interpolate_metric(
#                             cov, tpr
#                         )

#             # ---- Compute max TPR gap ----
#             for ii, split in enumerate(["train", "test"]):
#                 ax = axes[ii]

#                 all_delta = []
#                 for s in range(n_seeds):
#                     delta_seed = []
#                     for i, j in paired_subgroups(conditions[disparity]):
#                         gap = np.abs(
#                             interpolated_by_seed[s][split]["TPR_O"][i](desired_cov)
#                             - interpolated_by_seed[s][split]["TPR_O"][j](desired_cov)
#                         )
#                         delta_seed.append(gap)

#                     delta_seed = np.max(delta_seed, axis=0)
#                     all_delta.append(delta_seed)

#                 all_delta = np.array(all_delta)
#                 mean = np.mean(all_delta, axis=0)
#                 std = np.std(all_delta, axis=0)

#                 ax.plot(desired_cov, mean, label=method)
#                 ax.fill_between(
#                     desired_cov, mean - std, mean + std, alpha=0.2
#                 )

#                 ax.set_xlabel("Transparency")
#                 ax.set_title(split.capitalize())
#                 ax.grid(True)

#         axes[0].set_ylabel(
#             f"Maximum TPR Gap ({disparity})"
#         )

#         axes[1].legend(
#             loc="upper right", frameon=True
#         )

#         plt.suptitle(
#             f"Maximum TPR Disparity — {dataset_name.upper()} ({disparity})",
#             fontsize=14
#         )

#         plt.tight_layout(rect=[0, 0, 1, 0.93])

#         plt.savefig(
#             out_dir / f"MaxTPRGap_{dataset_name}_{disparity}.png",
#             dpi=300
#         )
#         plt.close()



#######################################

# TPR VS ICF , aggreaged for each demographic group 

######################################


# demographic_groups = ['Age', 'Gender', 'Race']

# for disparity in demographic_groups:

#     # reference group (most frequent)
#     freq = my_data.get_condition_freq(conditions[disparity])
#     reference = conditions[disparity][np.argmax(list(freq.values()))]

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

#     for i, split in enumerate(['train', 'test']):
#         ax = axes[i]

#         for cond in conditions[disparity]:
#             if cond == reference:
#                 continue
#             # --- ICF ---
#             ICF = np.array([
#                 interpolated_by_seed[s][split]['ICF'][cond](desired_cov)
#                 for s in range(n_seeds)
#             ])
#             mean_icf = np.mean(ICF, axis=0)
#             std_icf = np.std(ICF, axis=0)

#             # --- ΔTPR (global fairness) ---
#             delta_tpr = np.array([
#                 interpolated_by_seed[s][split]['TPR_O'][cond](desired_cov)
#                 - interpolated_by_seed[s][split]['TPR_O'][reference](desired_cov)
#                 for s in range(n_seeds)
#             ])
#             mean_delta = np.mean(delta_tpr, axis=0)
#             std_delta = np.std(delta_tpr, axis=0)

#             # --- scatter ---
#             sc = ax.scatter(
#                 mean_icf,
#                 mean_delta,
#                 c=desired_cov,
#                 cmap="viridis",
#                 s=40,
#                 alpha=0.8,
#                 label=cond
#             )

#             # --- uncertainty ---
#             ax.errorbar(
#                 mean_icf,
#                 mean_delta,
#                 xerr=std_icf,
#                 yerr=std_delta,
#                 fmt="none",
#                 ecolor="gray",
#                 alpha=0.15
#             )

#             # --- trend line per subgroup ---
#             if len(mean_icf) > 2:
#                 slope, intercept, r, p, _ = linregress(mean_icf, mean_delta)
#                 x_line = np.linspace(mean_icf.min(), mean_icf.max(), 100)
#                 y_line = slope * x_line + intercept
#                 ax.plot(
#                     x_line,
#                     y_line,
#                     linestyle="--",
#                     linewidth=1.5,
#                     alpha=0.7
#                 )

#         ax.axhline(0, color="red", linestyle=":", linewidth=1)
#         ax.set_title(f"{split.capitalize()}")
#         ax.set_xlabel("Interpretability Coverage Fairness (ICF)")
#         ax.grid(True)

#     axes[0].set_ylabel(f"TPR Disparity (vs. {reference})")

#     # shared colorbar
#     cbar = fig.colorbar(sc, ax=axes, shrink=0.85)
#     cbar.set_label("Transparency (coverage)")

#     # legend outside
#     handles, labels = axes[0].get_legend_handles_labels()
#     fig.legend(
#         handles,
#         labels,
#         title=f"{disparity} subgroups",
#         bbox_to_anchor=(1.02, 0.5),
#         loc="center left"
#     )

#     plt.tight_layout(rect=[0, 0, 0.85, 1])
#     file_path = out_dir / f"Global_Fairness_TPR_vs_ICF_{disparity}_{method}_{dataset_name}.png"
#     plt.savefig(file_path)
#     #plt.show()
#     plt.close()





# ##########################################

# #Global Fiarness :TPR Vs. ICF for each pairs of subgroup 
#Delta TPR VS Delta_ICF

# #########################################



# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:
#     for i,j in paired_subgroups(conditions[disparity]):
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         for ss, split in enumerate(['train', 'test']):
#             ax = axes[ss]
#             delta_tpr = np.array([interpolated_by_seed[s][split]['TPR_O'][i](desired_cov)- interpolated_by_seed[s][split]['TPR_O'][j](desired_cov) for s in range(n_seeds)])
#             mean_delta = np.mean(delta_tpr, axis=0)
#             std_delta = np.std(delta_tpr, axis=0)

            
#             delta_ICF = np.array([interpolated_by_seed[s][split]['ICF'][i](desired_cov)- interpolated_by_seed[s][split]['ICF'][j](desired_cov) for s in range(n_seeds)])
#             mean = np.mean(delta_ICF, axis=0)
#             std = np.std(delta_ICF, axis=0)
#             sc = ax.scatter(mean,mean_delta,c=desired_cov,cmap="viridis",s=40,alpha=0.8)
#             cbar = plt.colorbar(sc, ax=ax)
#             cbar.set_label("Transparency (coverage)")
#             ax.set_xlabel(f"ICF Difference ({i} - {j})")
#             ax.set_ylabel(f"TPR Difference ({i} - {j})")
#             ax.set_title(f'{split.capitalize()}')

#             #to fit a trend line and show Pearson correlation 
#             slope, intercept, r, p, _ = linregress(mean, mean_delta)
#             x_line = np.linspace(mean.min(), mean.max(), 100)
#             y_line = slope * x_line + intercept
            
#             rho, p = spearmanr(mean, mean_delta)
#             rho_abs, p_abs = spearmanr(np.abs(mean),np.abs(mean_delta))
#             #print(f"spearman correlation {rho:.2f} and p_value {p:.3f}")
#             ax.plot(x_line,y_line,linestyle="--",color="black",linewidth=2,label=f"Linear fit (r={r:.2f})")
#             ax.legend()

#             #to show uncertainty by faint error bar (with computed std)
#             ax.errorbar(mean,mean_delta,xerr=std,yerr=std_delta,fmt="none",ecolor="gray",alpha=0.3,capsize=0)
#             ax.axhline(0, color="red", linestyle=":", linewidth=1)


#         plt.tight_layout()
#         file_path = out_dir / f"DeltaICF_{i}_{j}_vs_DeltaTPR_{method}_{dataset_name}.png"
#         plt.savefig(file_path)
#         plt.show()
#         plt.close()

#second version with all pairs in one plot per demographic group
#-----

# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:
#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
#     axes_dict = {'train': axes[0], 'test': axes[1]}
    
#     for ss, split in enumerate(['train', 'test']):
#         ax = axes_dict[split]
#         for n, (i, j) in enumerate(paired_subgroups(conditions[disparity])):
#             if n>6:
#                 continue
#             # Compute delta TPR
#             delta_tpr = np.array([
#                 interpolated_by_seed[s][split]['TPR_O'][i](desired_cov) - 
#                 interpolated_by_seed[s][split]['TPR_O'][j](desired_cov) 
#                 for s in range(n_seeds)
#             ])
#             mean_delta_tpr = np.mean(delta_tpr, axis=0)
#             std_delta_tpr = np.std(delta_tpr, axis=0)

#             # Compute delta ICF
#             delta_ICF = np.array([
#                 interpolated_by_seed[s][split]['ICF'][i](desired_cov) - 
#                 interpolated_by_seed[s][split]['ICF'][j](desired_cov) 
#                 for s in range(n_seeds)
#             ])
#             mean_delta_ICF = np.mean(delta_ICF, axis=0)
#             std_delta_ICF = np.std(delta_ICF, axis=0)

#             # Scatter plot
#             sc = ax.scatter(
#                 mean_delta_ICF, mean_delta_tpr, c=desired_cov, cmap="viridis", 
#                 s=40, alpha=0.8
#             )

#             # Annotate mean point with pair name
#             mean_x = np.mean(mean_delta_ICF)
#             mean_y = np.mean(mean_delta_tpr)
#             ax.text(mean_x, mean_y, f"{i}-{j}", fontsize=6, weight='bold', 
#                     color='black', ha='right', va='bottom',)

#             # Trend line
#             slope, intercept, r, _, _ = linregress(mean_delta_ICF, mean_delta_tpr)
#             x_line = np.linspace(mean_delta_ICF.min(), mean_delta_ICF.max(), 100)
#             y_line = slope * x_line + intercept
#             ax.plot(x_line, y_line, linestyle="--", color="black", linewidth=1)

#             # Uncertainty bars
#             # ax.errorbar(mean_delta_ICF, mean_delta_tpr, xerr=std_delta_ICF, yerr=std_delta_tpr,
#             #             fmt="none", ecolor="gray", alpha=0.3, capsize=0)

#         # Axes labels, title, zero lines
#         ax.set_xlabel("ICF Difference")
#         ax.set_ylabel("TPR Difference")
#         ax.set_title(f"{split.capitalize()}")
#         ax.axhline(0, color="red", linestyle=":", linewidth=1)
#         ax.axvline(0, color="red", linestyle=":", linewidth=1)

#     # Colorbar for coverage
#     cbar = fig.colorbar(sc, ax=axes.ravel().tolist())
#     cbar.set_label("Transparency (coverage)")

#    #plt.suptitle(f"Delta ICF vs Delta TPR for {disparity}", fontsize=16)
#     plt.tight_layout(rect=[0, 0, 0.85, 0.95])

#     # Save figure
#     file_path = out_dir / f"DeltaICF_vs_DeltaTPR_{method}_{dataset_name}_{disparity}.png"
#     plt.savefig(file_path)
#     plt.show()
#     plt.close()



#-----




#################################

#Store All correlations from Delta ICF - Delta TPR analysis

################################

# all_results = []
# for m in experiments['method']:
#     for d in experiments['data']: 
#         dataset_name = d
#         method = m

        
#         # Load data
#         my_data = Dataset.from_csv(Path.cwd().parent/f'examples/data/{dataset_name}_mined.csv', dataset_name)
#         my_data.pre_process()

#         conditions = my_data.demographicGroup()
#         conds = conditions['All']


#         #read all results for one dataset and one method 
#         result_dir = Path.cwd()/'results'

#         results = load_results(dataset_name, method)


#         #group by all data per seed
#         by_seed = aggregate_results(results)


#         # Store interpolated functions per seed and split
#         interpolated_by_seed = {k: {'train': {}, 'test': {}} for k in range(n_seeds)}


#         for seed in by_seed:
#             for split in ['train', 'test']:
#                 cov = by_seed[seed][split]['cov']
                
#                 # Accuracy
#                 acc = by_seed[seed][split]['acc']
#                 interpolated_by_seed[seed][split]['acc'] = interpolate_metric(cov, acc) #fill_value= np.nan
                
#                 # ICF per condition
#                 interpolated_by_seed[seed][split]['ICF'] = {}
#                 for cond in conds:
#                     icf = by_seed[seed][split]['ICF'][cond]
#                     interpolated_by_seed[seed][split]['ICF'][cond] = interpolate_metric(cov, icf)
                
#                 # TPR metrics per condition
#                 for tpr_type in ['TPR_O', 'TPR_T', 'TPR_B']:
#                     interpolated_by_seed[seed][split][tpr_type] = {}
#                     for cond in conds:
#                         tpr = by_seed[seed][split][tpr_type][cond]
#                         interpolated_by_seed[seed][split][tpr_type][cond] = interpolate_metric(cov, tpr)
                
#                 # Coverage itself (just in case)
#                 interpolated_by_seed[seed][split]['cov'] = interpolate_metric(cov, cov)


#         min_cov = max(np.min(by_seed[s][split]['cov']) for s in range(n_seeds))
#         max_cov = min(np.max(by_seed[s][split]['cov']) for s in range(n_seeds))
#         desired_cov = np.linspace(min_cov, max_cov, 50)




#         demographic_groups = ['Age', 'Gender', 'Race']
#         for disparity in demographic_groups:
#             for i,j in paired_subgroups(conditions[disparity]):
#                 for ss, split in enumerate(['train', 'test']):
#                     delta_tpr = np.array([interpolated_by_seed[s][split]['TPR_T'][i](desired_cov)- interpolated_by_seed[s][split]['TPR_T'][j](desired_cov) for s in range(n_seeds)])
#                     mean_delta = np.mean(delta_tpr, axis=0)
#                     std_delta = np.std(delta_tpr, axis=0)
                    
#                     delta_ICF = np.array([interpolated_by_seed[s][split]['ICF'][i](desired_cov)- interpolated_by_seed[s][split]['ICF'][j](desired_cov) for s in range(n_seeds)])
#                     mean = np.mean(delta_ICF, axis=0)
#                     std = np.std(delta_ICF, axis=0)

#                     rho, p = spearmanr(mean, mean_delta)
#                     rho_abs, p_abs = spearmanr(np.abs(mean),np.abs(mean_delta))



#                     all_results.append({
#                     "dataset": dataset_name,
#                     "method": method,
#                     "demographic": disparity,
#                     "pair": f"{i}-{j}",
#                     "split": split,
#                     "rho": rho,  #spearman correlation
#                     "p": p, #p-value
#                     "rho_abs": rho_abs, #spearman correlation for absolute values
#                     "p_abs": p_abs}) #p-value for absolute values



# save_json(all_results, "correaltion_T.json")


##############################################
#Analysis of correlation results
##############################################

# corr = read_json('correaltion_B.json')


# df = pd.DataFrame(corr)
# df["abs_rho"] = np.abs(df["rho"])
# df["significant"] = df["p"] < 0.05
# df["positive"] = df["rho"] > 0
# df["negative"] = df["rho"] < 0


# #analysis of corr(delta_ICF, delta_TPR):
# summary = []
# for method in df["method"].unique():
#     for demo in df["demographic"].unique():
#         for data in df["dataset"].unique():

#             sub = df[
#                 (df["method"] == method) &
#                 (df["demographic"] == demo) &
#                 (df["dataset"] == data) &
#                 (df["split"] == "test")
#             ]

#             if len(sub) == 0:
#                 continue

#             n = len(sub)

#             pct_abs = np.mean(sub["abs_rho"] >= 0.3) * 100
#             pct_sig = np.mean(sub["significant"]) * 100
#             pct_pos = np.mean(sub["positive"]) * 100
#             pct_neg = np.mean(sub["negative"]) * 100
#             med_rho = np.median(sub["rho"])
#             med_rho_abs = np.median(sub["abs_rho"])

#             # Determine dominant direction
#             if abs(pct_pos - pct_neg) < 20:
#                 pattern = "mixed"
#             elif pct_pos > pct_neg:
#                 pattern = "positive"
#             else:
#                 pattern = "negative"

#             summary.append({
#                 "Method": method,
#                 "Demographic": demo,
#                 "Dataset": data,
#                 "N pairs": n,
#                 "% |ρ| ≥ 0.3": round(pct_abs, 1),
#                 "% significant": round(pct_sig, 1),
#                 "Median ρ": round(med_rho, 3),
#                 "Median |ρ|": round(med_rho_abs, 3),
#                 "% positive": round(pct_pos, 1),
#                 "% negative": round(pct_neg, 1),
#                 "Dominant pattern": pattern
#             })

# summary_df = pd.DataFrame(summary)

# # Optional: nicer ordering
# summary_df = summary_df.sort_values(
#     ["Method", "Demographic"]
# ).reset_index(drop=True)

# print(summary_df)


# # Save summary to CSV
# summary_df.to_csv("correlation_summary.csv", index=False)


# #analysis of corr(abs(delta_ICF), abs(delta_TPR)):
# df = pd.DataFrame(corr)
# df["abs_rho"] = np.abs(df["rho_abs"])
# df["significant"] = df["p_abs"] < 0.05
# df["positive"] = df["rho_abs"] > 0
# df["negative"] = df["rho_abs"] < 0

# summary = []
# for method in df["method"].unique():
#     for demo in df["demographic"].unique():
#         for data in df["dataset"].unique():

#             sub = df[
#                 (df["method"] == method) &
#                 (df["demographic"] == demo) &
#                 (df["dataset"] == data) &
#                 (df["split"] == "test")
#             ]

#             if len(sub) == 0:
#                 continue

#             n = len(sub)

#             pct_abs = np.mean(sub["abs_rho"] >= 0.3) * 100
#             pct_sig = np.mean(sub["significant"]) * 100
#             pct_pos = np.mean(sub["positive"]) * 100
#             pct_neg = np.mean(sub["negative"]) * 100
#             med_rho = np.median(sub["rho_abs"])
#             med_rho_abs = np.median(sub["abs_rho"])

#             # Determine dominant direction
#             if abs(pct_pos - pct_neg) < 20:
#                 pattern = "mixed"
#             elif pct_pos > pct_neg:
#                 pattern = "positive"
#             else:
#                 pattern = "negative"

#             summary.append({
#                 "Method": method,
#                 "Demographic": demo,
#                 "Dataset": data,
#                 "N pairs": n,
#                 "% |ρ| ≥ 0.3": round(pct_abs, 1),
#                 "% significant": round(pct_sig, 1),
#                 "Median ρ": round(med_rho, 3),
#                 "Median |ρ|": round(med_rho_abs, 3),
#                 "% positive": round(pct_pos, 1),
#                 "% negative": round(pct_neg, 1),
#                 "Dominant pattern": pattern
#             })

# summary_df = pd.DataFrame(summary)

# # Optional: nicer ordering
# summary_df = summary_df.sort_values(
#     ["Method", "Demographic"]
# ).reset_index(drop=True)

# print(summary_df)

# # Save summary to CSV
# summary_df.to_csv("correlation_abs_summary.csv", index=False)   



##############################################
#Plot proportion of positive lables for each pair of subgropus
##############################################

# demographic_groups = ['Age', 'Gender', 'Race']
# for disparity in demographic_groups:
#     for k in paired_subgroups(conditions[disparity]):
#         fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#         for s,split in enumerate(['train', 'test']):
#             for i in k:
#                 ax = axes[s]
#                 pos_ratio = {k:[] for k in ['T','B']}
#                 pos_ratio['T'] = [interpolated_by_seed[s][split]['pos_ratio_T'][i](desired_cov) for s in range(n_seeds)]
#                 pos_ratio['B'] = [interpolated_by_seed[s][split]['pos_ratio_B'][i](desired_cov) for s in range(n_seeds)]

#                 mean = {k:np.mean(pos_ratio[k], axis=0) for k in ['T','B']}

#                 std = {k:np.std(pos_ratio[k], axis=0) for k in ['T','B']}


#                 styles = {'T': dict(lw=1.5, ls='--', alpha=0.8),
#                         'B': dict(lw=1.5, ls=':', alpha=0.8),}
#                 for pos_type in ['T','B']:
#                     ax.plot(desired_cov, mean[pos_type],
#                             label = {'T': f'Transparent Part {i}','B': f'Black-box Part {i}'}[pos_type],
#                             **styles[pos_type]) 
                

#                     ax.set_xlabel("Transparency")
#                     ax.set_ylabel(f"Proportion of Positive Labels for {k[0]},{k[1]}")
#                     ax.set_title(f"{split.capitalize()}")

#                     ax.legend()
#                     ax.grid(True)
#                     ax.axhline(0, color='black', lw=1, alpha=0.6)
#         plt.tight_layout()
#         file_path = out_dir / f"Signed_Disparity_{i}_{method}_{dataset_name}.png"
#         #plt.savefig(file_path)
#         plt.show()
#         plt.close()



# demographic_groups = ['Age', 'Gender', 'Race']

# for disparity in demographic_groups:
#     for k in paired_subgroups(conditions[disparity]):

#         fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

#         # Assign a unique color to each subgroup in this pair
#         colors = dict(zip(
#             k,
#             plt.cm.tab10.colors[:len(k)]  # safe for up to 10 subgroups
#         ))

#         for s, split in enumerate(['train', 'test']):
#             ax = axes[s]

#             for i in k:
#                 pos_ratio = {}

#                 pos_ratio['T'] = np.array([
#                     interpolated_by_seed[seed][split]['pos_ratio_T'][i](desired_cov)
#                     for seed in range(n_seeds)
#                 ])
#                 pos_ratio['B'] = np.array([
#                     interpolated_by_seed[seed][split]['pos_ratio_B'][i](desired_cov)
#                     for seed in range(n_seeds)
#                 ])

#                 mean = {p: np.mean(pos_ratio[p], axis=0) for p in ['T', 'B']}
#                 std  = {p: np.std(pos_ratio[p], axis=0)  for p in ['T', 'B']}

#                 styles = {
#                     'T': dict(ls=':', lw=2),
#                     'B': dict(  lw=2),
#                 }

#                 for pos_type in ['T', 'B']:
#                     ax.plot(
#                         desired_cov,
#                         mean[pos_type],
#                         color=colors[i],                # subgroup color
#                         label=f"{i} - {'T' if pos_type=='T' else 'B'}",
#                         alpha=0.9,
#                         **styles[pos_type]
#                     )

#             ax.set_xlabel("Transparency")
#             ax.set_title(f"{split.capitalize()}")
#             ax.grid(True)
#             ax.axhline(0, color='black', lw=1, alpha=0.6)

#         axes[0].set_ylabel(
#             f"Proportion of Positive Labels\n(subgroups: {k[0]} vs {k[1]})"
#         )

#         # One legend for the whole figure
#         handles, labels = axes[0].get_legend_handles_labels()
#         fig.legend(handles, labels, loc="center right")

#         plt.tight_layout(rect=[0, 0, 0.85, 1])

#         file_path = out_dir / f"Pos_ratio{k[0]}_{k[1]}_{method}_{dataset_name}.png"
#         plt.savefig(file_path)
#         #plt.show()
#         plt.close()




############################################

#This plot Delta TPR and Positive ratio together for each pair of subgroups

############################################
demographic_groups = ['Age', 'Gender', 'Race']

for disparity in demographic_groups:
    for i, j in paired_subgroups(conditions[disparity]):

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10),
            sharex=True, constrained_layout=True
        )

        for col, split in enumerate(['train', 'test']):

            # =======================
            # (TOP) POSITIVE RATIO
            # =======================
            ax = axes[0, col]

            colors = {i: 'tab:blue', j: 'tab:orange'}
            styles = {'T': ':', 'B': '-'}

            for subgroup in [i, j]:
                for pos_type in ['T', 'B']:
                    pos_ratio = np.array([
                        interpolated_by_seed[s][split][
                            'pos_ratio_T' if pos_type == 'T' else 'pos_ratio_B'
                        ][subgroup](desired_cov)
                        for s in range(n_seeds)
                    ])

                    mean = np.mean(pos_ratio, axis=0)

                    ax.plot(
                        desired_cov,
                        mean,
                        color=colors[subgroup],
                        linestyle=styles[pos_type],
                        lw=2,
                        label=f"{subgroup} - {pos_type}"
                    )

            ax.set_title(f"{split.capitalize()} — Difficulty Routing")
            ax.set_ylabel("Proportion of Positive Labels",  labelpad=12)
            ax.grid(True)

            # =======================
            # (BOTTOM) ΔTPR
            # =======================
            ax = axes[1, col]

            delta_tpr_all = {k: [] for k in ['TPR_O', 'TPR_T', 'TPR_B']}

            for s in range(n_seeds):
                for tpr_type in ['TPR_O', 'TPR_T', 'TPR_B']:
                    delta = (
                        interpolated_by_seed[s][split][tpr_type][i](desired_cov)
                        - interpolated_by_seed[s][split][tpr_type][j](desired_cov)
                    )
                    delta_tpr_all[tpr_type].append(delta)

            for tpr_type, style in {
                'TPR_O': dict(lw=2.5),
                'TPR_T': dict(ls='--', alpha=0.9),
                'TPR_B': dict(ls=':', alpha=0.9),
            }.items():

                mean = np.mean(np.array(delta_tpr_all[tpr_type]), axis=0)
                ax.plot(
                    desired_cov,
                    mean,
                    label={
                        'TPR_O': 'Overall',
                        'TPR_T': 'Transparent',
                        'TPR_B': 'Black-box'
                    }[tpr_type],
                    **style
                )

            ax.axhline(0, color='black', lw=1)
            ax.set_xlabel("Transparency")
            ax.set_ylabel(f"ΔTPR ({i} − {j})",  labelpad=12)
            ax.set_title(f"{split.capitalize()} — Outcome Disparity")
            ax.grid(True)

        # Legend for TOP row (difficulty routing)
        top_handles, top_labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            top_handles, top_labels,
            loc='upper right',
            bbox_to_anchor=(1.02, 0.85),
            title="Routing (subgroup × component)"
        )

        # Legend for BOTTOM row (ΔTPR decomposition)
        bottom_handles, bottom_labels = axes[1, 0].get_legend_handles_labels()
        fig.legend(
            bottom_handles, bottom_labels,
            loc='lower right',
            bbox_to_anchor=(1.02, 0.15),
            title="ΔTPR Decomposition"
)



        file_path = out_dir/"composit" / f"Composite_{i}_{j}_{method}_{dataset_name}.png"
        plt.savefig(file_path)
        #plt.show()
        plt.close()

