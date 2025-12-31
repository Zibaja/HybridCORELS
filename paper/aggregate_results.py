import json
import glob
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("results")
OUT_FILE = Path("aggregated_results.json")


def dd():
    return defaultdict(dd)


ALL_RESULTS = dd()


def merge_trans(dst, src):
    for split in ["train", "test"]:
        for cond in src[split]:
            for k in ["ICF", "TP", "FP", "TN", "FN"]:
                if k == "ICF":
                    dst[split][cond]["ICF"].extend(src[split][cond]["ICF"])
                else:
                    for s in ["T", "B"]:
                        dst[split][cond][k][s].extend(
                            src[split][cond][k][s]
                        )


for f in glob.glob(str(RESULTS_DIR / "*.json")):
    with open(f) as fh:
        r = json.load(fh)

    block = ALL_RESULTS[r["dataset"]][r["model"]][str(r["min_coverage"])]

    for split in ["train", "test"]:
        block.setdefault("accuracy", {}).setdefault(split, []).append(
            r["accuracy"][split]
        )
        block.setdefault("coverage", {}).setdefault(split, []).append(
            r["coverage"][split]
        )

    block.setdefault("trans_total", r["trans_total"])
    merge_trans(block["trans_total"], r["trans_total"])


with open(OUT_FILE, "w") as f:
    json.dump(ALL_RESULTS, f, indent=2)

#print(f"Aggregated results written to {OUT_FILE}")


