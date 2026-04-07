
from pathlib import Path
from exp_utils import read_json, Dataset, paired_subgroups
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d

result_dir = Path.cwd()/'results'
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


experiments = {'data': ["compas", "adult", "acs_employ"], 
               'method':["HybridCORELSPreClassifier","HybridCORELSPostClassifier", "CRL","HyRS"]
               , 'tardeoff_value':TRADEOFF_VALUES.values(),'seed': list(range(n_seeds))}


dataset_name = 'compas'
method = 'CRL'
seed = 0

all_dirs = [i for i in result_dir.iterdir() if dataset_name in i.stem and method in i.stem and str(seed) in i.stem]





def load_results(dataset, method):
    results = []

    for f in result_dir.glob(f"{dataset}__{method}__*.json"):
        r = read_json(f)
        results.append(r)

    return results


results = load_results(dataset_name, method)
accuracy_list = [i['accuracy']['train'] for i in results ]
accuracy_list_flat = [j for i in accuracy_list for j in i]

cov_list =  [i['coverage']['train'] for i in results ]
cov_list_flat = [j for i in cov_list for j in i]



def get_TPR_one_data (data, split):
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
        'param': []
    }
conds = list(results[0]['trans_total']['train'].keys())
by_seed = {seed: {'train': make_split_dict(conds), 'test': make_split_dict(conds)} for seed in range(n_seeds)}

TRADEOFF_PARAM = {
    "HybridCORELSPreClassifier": "min_coverage",
    "HybridCORELSPostClassifier": "min_coverage",
    "CRL": "alpha",
    "HyRS": "beta",
}

def aggregate_results (results):
    by_seed = {seed: {'train': make_split_dict(conds), 'test': make_split_dict(conds)} for seed in range(n_seeds)}

    for r in results:
        seed = r["seed"]
        TPR = {split: get_TPR_one_data(r, split) for split in ['train', 'test']}

        for split in ['train', 'test']:
            by_seed[seed][split]['cov'].extend(r['coverage'][split])
            by_seed[seed][split]['acc'].extend(r['accuracy'][split])
            
            for tpr_type, key in zip(['overall', 'T', 'B'], ['TPR_O', 'TPR_T', 'TPR_B']):
                for cond in conds:
                    by_seed[seed][split][key][cond].extend(TPR[split][cond][tpr_type])
            
            for cond in conds:
                by_seed[seed][split]['ICF'][cond].extend(r['trans_total'][split][cond]['ICF'])
            
            by_seed[seed][split]['param'].append(r[TRADEOFF_PARAM[r['model']]])
    
    return by_seed

by_seed = aggregate_results(results)


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

   

    # Group by coverage
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
    


# Store interpolated functions per seed and split
interpolated_by_seed = {k: {'train': {}, 'test': {}} for k in range(10)}

for seed in by_seed:
    for split in ['train', 'test']:
        cov = by_seed[seed][split]['cov']
        
        # Accuracy
        acc = by_seed[seed][split]['acc']
        interpolated_by_seed[seed][split]['acc'] = interpolate_metric(cov, acc)
        
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
        
        # Coverage itself (just in case)
        interpolated_by_seed[seed][split]['cov'] = interpolate_metric(cov, cov)


desired_cov = np.linspace(0, 1, 50)  # 50 points from 0 to 1

all_acc = np.array([interpolated_by_seed[s]['train']['acc'](desired_cov) for s in range(n_seeds)])

mean_acc = np.mean(all_acc, axis=0)
std_acc = np.std(all_acc, axis=0)









