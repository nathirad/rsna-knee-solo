"""Gold-58 comparison of the sources we can submit or mix: the public 0.943 ensemble ("base"),
our own 5-fold model ("solo") and the public M448 reader.

  per-label AUC of each source, macro AUC with a bootstrap interval
  rank correlation between sources (how much new information a mix can bring)
  rank-mix curves base + X at several weights, with a paired bootstrap against base alone

Caveat printed with the numbers: the base ensemble and M448 were trained with the gold labels
in their training data, so their gold-58 AUC is an upper bound, and a mix that looks worse
than base here is partly an artefact of that. Our solo model never saw gold-58.

Usage (from the RSNA project folder):
  python research/gold_blend_check.py [--base runs/rsna-knee-base-gold/latest/submission.csv]
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
WEIGHTS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
N_BOOT = 2000


def load(path):
    return pd.read_csv(path, dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


def auc(y, p):
    """AUC from average ranks (Mann-Whitney), ties counted as half; same value as sklearn's."""
    n1 = int(y.sum()); n0 = len(y) - n1
    if not n1 or not n0:
        return np.nan
    return (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(y, p, rows=None):
    """Macro AUC; a label with one class in the resample is skipped."""
    if rows is not None:
        y, p = y[rows], p[rows]
    out = [auc(y[:, j], p[:, j]) for j in range(y.shape[1])]
    return float(np.nanmean(out)), out


def mix(a, b, w):
    """The rank mix the submission notebooks use."""
    return ((1 - w) * a.rank(pct=True) + w * b.rank(pct=True)).rank(pct=True)


ap = argparse.ArgumentParser()
ap.add_argument("--train-csv", default="data/train.csv")
ap.add_argument("--base", default="runs/rsna-knee-base-gold/latest/submission.csv")
ap.add_argument("--solo-glob", default="runs/rsna-knee-train-solo-*/latest/res/gold_cnx_224_pilk_one_f*_s0.csv")
ap.add_argument("--m448", default="runs/rsna-knee-m448-check/latest/m448_train_check.csv")
ap.add_argument("--extra", nargs="*", default=[], help="name=path of further prediction files on the gold studies")
ap.add_argument("--show-w", type=float, default=0.10, help="weight of the per-label table")
ap.add_argument("--legs", nargs="*", default=["solo", "m448", "solo+m448"], help="legs to mix into base; x+y is a rank-mean")
a = ap.parse_args()

gold = pd.read_csv(a.train_csv, dtype={"StudyInstanceUID": str}).dropna(subset=T).set_index("StudyInstanceUID")[T].astype(int)
y = gold.values
print(f"gold studies: {len(gold)}; positives per label: " + ", ".join(f"{t} {int(gold[t].sum())}" for t in T))

src = {}
files = sorted(glob.glob(a.solo_glob))
if files:
    src["solo"] = sum(load(f).loc[gold.index, T].rank(pct=True) for f in files) / len(files)
    print(f"solo: rank-mean of {len(files)} fold models")
if Path(a.m448).is_file():
    src["m448"] = load(a.m448).loc[gold.index, T]
if Path(a.base).is_file():
    b = load(a.base)
    missing = gold.index.difference(b.index)
    assert not len(missing), f"base file lacks {len(missing)} gold studies"
    src["base"] = b.loc[gold.index, T]
for e in a.extra:
    name, path = e.split("=", 1)
    src[name] = load(path).reindex(gold.index)[T]
    if src[name].isna().any().any():
        print(f"{name}: {int(src[name].isna().any(axis=1).sum())} gold studies without a prediction, filled with the column median")
        src[name] = src[name].fillna(src[name].median())

rng = np.random.default_rng(0)
boots = [rng.integers(0, len(gold), len(gold)) for _ in range(N_BOOT)]

print("\n== per-label AUC on gold-58 ==")
tab = pd.DataFrame({k: macro(y, v.values)[1] for k, v in src.items()}, index=T)
tab.loc["MACRO"] = tab.mean()
print(tab.round(3).to_string())
for k, v in src.items():
    d = np.array([macro(y, v.values, r)[0] for r in boots])
    print(f"{k}: macro {macro(y, v.values)[0]:.4f}, 95% interval [{np.percentile(d, 2.5):.3f}, {np.percentile(d, 97.5):.3f}]")

print("\n== Spearman correlation between sources (mean over the 12 labels) ==")
names = list(src)
for i, p in enumerate(names):
    for q in names[i + 1:]:
        c = [src[p][t].corr(src[q][t], method="spearman") for t in T]
        print(f"{p} ~ {q}: {np.mean(c):.3f}  (min {np.min(c):.3f} {T[int(np.argmin(c))]}, max {np.max(c):.3f} {T[int(np.argmax(c))]})")

anchor = "base" if "base" in src else None
if anchor:
    print(f"\n== rank mix {anchor} + X: macro AUC on gold-58, paired bootstrap vs {anchor} alone ==")
    a0 = macro(y, src[anchor].values)[0]
    legs = {}                                                           # the legs we could add
    for spec in a.legs:                                                 # "x+y" = rank-mean of x and y
        parts = [q for q in spec.split("+") if q in src]
        if len(parts) == len(spec.split("+")):
            legs[spec] = sum(src[q].rank(pct=True) for q in parts) / len(parts)
    print("legs alone: " + ", ".join(f"{k} {macro(y, v.values)[0]:.4f}" for k, v in legs.items()))
    for k, v in legs.items():
        for w in WEIGHTS:
            m = mix(src[anchor], v, w).values
            d = np.array([macro(y, m, r)[0] - macro(y, src[anchor].values, r)[0] for r in boots])
            print(f"{anchor} + {k:16s} w={w:.2f}: {macro(y, m)[0]:.4f}  ({macro(y, m)[0] - a0:+.4f}, "
                  f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], P(better) {(d > 0).mean():.2f})")
    print(f"\nper-label change at w={a.show_w:.2f}:")
    ch = pd.DataFrame({k: np.array(macro(y, mix(src[anchor], v, a.show_w).values)[1]) - np.array(macro(y, src[anchor].values)[1])
                       for k, v in legs.items()}, index=T)
    print(ch.round(3).to_string())
