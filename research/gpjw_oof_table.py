"""Out-of-fold table of the public 2.5D reader recipe, round 1, and what it says about a mixed-label round.

Every train study gets the prediction of the one fold model that did not train on it:
folds 0-2 from the public checkpoints (run on their own folds through our cache), folds 3-4 from
our own `ctrl` models. Written to runs/solo/gpjw_oof_round1.csv (the input of a mixed-label round).

Printed:
  - out-of-fold AUC per fold and overall, against the old solo model on the same studies
  - gold-58 of the five models, and of the label table
  - gold-58 of the mixed target a*model + (1-a)*label, per finding: is the mix closer to the
    radiologists than the labels alone? (on gold the "model" is one fold model, as on train studies)

Usage (from the RSNA project folder): python research/gpjw_oof_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
R1, R2 = Path("runs/rsna-knee-gpjw-train-r1/latest/res"), Path("runs/rsna-knee-gpjw-train-r2/latest/res")
SRC = {0: R2 / "oof_public_fold0.csv", 1: R2 / "oof_public_fold1.csv", 2: R2 / "oof_public_fold2.csv",
       3: R1 / "oof_ctrl_fold3.csv", 4: R2 / "oof_ctrl_fold4.csv"}
OUT = Path("runs/solo/gpjw_oof_round1.csv")
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(y, p, ok=None, rows=None):
    if rows is not None:
        y, p, ok = y[rows], p[rows], (None if ok is None else ok[rows])
    out = [auc(y[:, j] if ok is None else y[ok[:, j], j], p[:, j] if ok is None else p[ok[:, j], j]) for j in range(y.shape[1])]
    return float(np.nanmean(out)), out


ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
lab = ld("runs/solo/gpjw_labels_v0.csv")
gold_ids = lab.index[lab.is_gold == 1]
gy = lab.loc[gold_ids, [f"gold_{t}" for t in T]].values.astype(int)
parts, gold_pred = [], {}
for k, path in SRC.items():
    d = ld(path)
    v = d[d.set == "val"][T]
    assert (lab.loc[v.index, "fold"] == k).all(), f"{path}: a study is not in fold {k}"
    v = v.assign(fold=k)
    parts.append(v)
    gold_pred[k] = d[d.set == "gold"].loc[gold_ids, T]
oof = pd.concat(parts)
assert oof.index.is_unique and len(oof) == int((lab.is_gold == 0).sum()), "the table does not cover every non-gold study once"
OUT.parent.mkdir(parents=True, exist_ok=True)
oof.to_csv(OUT)
print(f"saved {OUT}: {len(oof)} studies, folds {oof.fold.value_counts().sort_index().to_dict()}")

# ---- out-of-fold AUC against report labels ----
ids = oof.index
cy = (lab.loc[ids, T].values > 0.5).astype(int)
sl4 = pd.read_csv("research/public_notebooks/knee_g2/lab2/llm_labels_v4_blend.csv", dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].reindex(ids)
sy, sok = (sl4.values > 0.5).astype(int), (2 * sl4 - 1).abs().values >= 0.5
solo = ld("runs/solo/oof_round1.csv").reindex(ids)[T]
P, S = oof[T].values, solo.values
print("\n== out-of-fold AUC vs report labels (each study scored by a model that did not train on it) ==")
print(f"{'':10s} {'recipe: consensus>0.5':>22s} {'recipe: sl4 cells':>18s} {'old solo: sl4 cells':>20s}")
for k in range(5):
    r = np.where(oof.fold.values == k)[0]
    print(f"fold {k} ({'public' if k < 3 else 'ours  '}) {macro(cy, P, None, r)[0]:14.4f} {macro(sy, P, sok, r)[0]:18.4f} {macro(sy, S, sok, r)[0]:20.4f}")
m_c, m_s, m_o = macro(cy, P)[0], macro(sy, P, sok)[0], macro(sy, S, sok)[0]
rng = np.random.default_rng(0)
d = np.array([macro(sy, P, sok, r)[0] - macro(sy, S, sok, r)[0] for r in (rng.integers(0, len(ids), len(ids)) for _ in range(500))])
print(f"all folds  {m_c:20.4f} {m_s:18.4f} {m_o:20.4f}   recipe - old solo = {m_s - m_o:+.4f}, 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
per = pd.DataFrame({"recipe OOF": macro(sy, P, sok)[1], "old solo OOF": macro(sy, S, sok)[1]}, index=T)
per["diff"] = per["recipe OOF"] - per["old solo OOF"]
print(per.round(3).to_string())

# ---- gold-58 ----
L = lab.loc[gold_ids, T].values
ens = sum(v.values for v in gold_pred.values()) / 5
single = np.mean([macro(gy, v.values)[1] for v in gold_pred.values()], axis=0)
g = pd.DataFrame({"labels": macro(gy, L)[1], "one fold model (mean of 5)": single, "5 fold models together": macro(gy, ens)[1]}, index=T)
print("\n== gold-58: the labels, one fold model, all five ==")
print(pd.concat([g, g.mean().to_frame("MACRO").T]).round(3).to_string())
print("per model: " + ", ".join(f"f{k} {macro(gy, v.values)[0]:.4f}" for k, v in gold_pred.items()))

# ---- would a mixed target be closer to the radiologists than the labels? ----
print("\n== gold-58 AUC of the mixed target a*model + (1-a)*label (model = one fold model; mean over the 5) ==")
rows = {}
for a in ALPHAS:
    rows[f"a={a:.2f}"] = np.mean([macro(gy, a * v.values + (1 - a) * L)[1] for v in gold_pred.values()], axis=0)
mix = pd.DataFrame(rows, index=T)
mix["best a"] = [ALPHAS[int(np.argmax(mix.loc[t].values))] for t in T]
print(pd.concat([mix, mix.drop(columns="best a").mean().to_frame("MACRO").T]).round(3).to_string())
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(gy), len(gy)) for _ in range(2000)]
for a in (0.25, 0.5, 0.75):
    d = np.array([np.mean([macro(gy, a * v.values + (1 - a) * L, None, r)[0] for v in gold_pred.values()]) - macro(gy, L, None, r)[0] for r in boots])
    print(f"a={a:.2f} vs labels alone: {d.mean():+.4f}, 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], P(better) {(d > 0).mean():.2f}")
