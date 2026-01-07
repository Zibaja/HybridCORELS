
from pathlib import Path
from exp_utils import read_json

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



from pathlib import Path
from exp_utils import read_json
import numpy as np
from collections import defaultdict


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


def group_by_seed(results, split="test"):
    """
    Returns:
    seed → list of (coverage, accuracy)
    """
    by_seed = defaultdict(list)

    for r in results:
        seed = r["seed"]

        cov = r["coverage"][split]
        acc = r["accuracy"][split]

        # CRL case: lists
        if isinstance(cov, list):
            for c, a in zip(cov, acc):
                by_seed[seed].append((c, a))
        else:
            by_seed[seed].append((cov, acc))

    return by_seed


by_seed = group_by_seed(results, split = 'train')

print(by_seed)





