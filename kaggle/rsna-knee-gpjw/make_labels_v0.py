"""Label table of the public 2.5D reader (goodpjw2008 `make_labels.py`, version v0), rebuilt locally.

Soft target = mean of four public report-label tables, clipped to 0.01-0.99. Folds = GroupKFold(5)
on the normalised report text over the non-gold studies, with the same permutation seed, so the
folds match the public fold checkpoints (checked in runs/LOG.md, 2026-10-05). Gold studies keep
fold -1 and are never trained on.

Usage (from the RSNA project folder): python kaggle/rsna-knee-gpjw/make_labels_v0.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

RSNA = Path(__file__).resolve().parents[2]
OUT = RSNA / "runs/solo/gpjw_labels_v0.csv"
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SRC = {"steven": RSNA / "research/public_notebooks/knee_g2/lab2/llm_labels_v4_blend.csv",
       "pilkwang": RSNA / "research/public_notebooks/knee_g2/lab1/report_labels_v2.csv",
       "lixin": RSNA / "data/labels/labels_llm_gpt56sol.csv",
       "qwen": RSNA / "data/labels/qwen_knee_weak_labels.csv"}

tr = pd.read_csv(RSNA / "data/train.csv", dtype={"StudyInstanceUID": str})
ids = tr.StudyInstanceUID.values
gold = tr.set_index("StudyInstanceUID")[LABELS]
is_gold = gold.notna().all(axis=1).values
src = {}
for k, path in SRC.items():
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if k == "qwen":
        d = d[["StudyInstanceUID"] + [f"{c}__label" for c in LABELS]].set_axis(["StudyInstanceUID"] + LABELS, axis=1)
    src[k] = d.set_index("StudyInstanceUID")[LABELS].astype(float).reindex(ids)
    print(f"{k}: {int(src[k].notna().all(axis=1).sum())} of {len(ids)} studies fully labelled")
num = sum(src[k].fillna(0) for k in src)
den = sum(src[k].notna() * 1.0 for k in src)
soft = (num / den).clip(0.01, 0.99)
assert soft.notna().all().all(), "a study has no label source"

t = pd.DataFrame({k: [roc_auc_score(gold[c].values[is_gold], v[c].fillna(0.5).values[is_gold]) for c in LABELS]
                  for k, v in {**src, "consensus": soft}.items()}, index=LABELS)
t.loc["MEAN"] = t.mean()
print("AUC on the 58 gold studies (label source vs official label):")
print(t.round(3).to_string())

out = soft.copy()
out.insert(0, "is_gold", is_gold.astype(int))
out["fold"] = -1
weak = ~is_gold
groups = tr.Report.map(lambda s: re.sub(r"\W+", "", str(s).lower())).values[weak]
order = np.random.RandomState(0).permutation(weak.sum())
folds = np.zeros(weak.sum(), int)
for k, (_, va) in enumerate(GroupKFold(5).split(order, groups=groups[order])):
    folds[order[va]] = k
out.loc[weak, "fold"] = folds
for c in LABELS:
    out[f"gold_{c}"] = gold[c].values
out.index.name = "StudyInstanceUID"
OUT.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(OUT)
prev = pd.read_csv(RSNA / "runs/solo/gpjw_folds_reproduced.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").gpjw_fold
assert (prev.reindex(out.index) == out.fold).all(), "folds differ from the split checked against the public checkpoints"
print("saved", OUT, out.shape, "| fold sizes", out.fold.value_counts().sort_index().to_dict())
print("mean soft label:", out[LABELS].mean().round(2).to_dict())
