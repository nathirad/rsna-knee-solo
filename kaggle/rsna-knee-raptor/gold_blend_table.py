"""Gold-58 table for the public-single-model track: the two public CoAtNets (plain view, own heads),
our reader with calFG heads, and fixed-weight rank blends of them, each with a paired bootstrap against
the 0.949 model alone. Gold-58 is a pre-filter only: it has overstated what blends add (see runs/LOG.md);
the leaderboard decides.

Usage (from the RSNA project folder): python kaggle/rsna-knee-raptor/gold_blend_table.py
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(y, p, rows=None):
    if rows is not None:
        y, p = y[rows], p[rows]
    return np.array([auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


rk = lambda a: np.apply_along_axis(rankdata, 0, a) / len(a)
lab = ld("runs/solo/gpjw_labels_v0.csv")
gid = lab.index[lab.is_gold == 1]
y = lab.loc[gid, [f"gold_{t}" for t in T]].values.astype(int)


def raptor(run):
    z = np.load(f"runs/{run}/latest/res/feats_raptor_train.npz")
    st = [str(s) for s in z["studies"]]
    return z["logit"][[st.index(s) for s in gid]]


A, B = raptor("rsna-knee-raptor-feats-check"), raptor("rsna-knee-raptor224-gold")
g = lambda p: (lambda d: d[d.set == "gold"].loc[gid, T].values)(ld(p))
R = sum(g(f"runs/rsna-knee-gpjw-heads-a/latest/res/oof_calFG_fold{k}.csv") for k in range(5)) / 5
mix = lambda parts: sum(w * rk(p) for w, p in parts)
rows = {"A: public CoAtNet 384 (LB 0.949)": A, "B: public CoAtNet 224 (LB 0.945)": B, "R: our reader, calFG heads": R,
        "A 0.7 + R 0.3": mix([(0.7, A), (0.3, R)]), "A 0.85 + R 0.15": mix([(0.85, A), (0.15, R)]),
        "A 0.7 + B 0.3": mix([(0.7, A), (0.3, B)]), "A 0.5 + B 0.5": mix([(0.5, A), (0.5, B)]),
        "A 0.5 + B 0.2 + R 0.3": mix([(0.5, A), (0.2, B), (0.3, R)]), "A 0.4 + B 0.3 + R 0.3": mix([(0.4, A), (0.3, B), (0.3, R)])}
tab = pd.DataFrame({k: per(y, v) for k, v in rows.items()}, index=T).T
tab.insert(0, "MACRO", tab.mean(axis=1))
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(3000)]
ref = rk(A)
for k in list(rows)[3:]:
    d = np.array([np.nanmean(per(y, rows[k], b)) - np.nanmean(per(y, ref, b)) for b in boots])
    tab.loc[k, "vs A"], tab.loc[k, "CI low"], tab.loc[k, "CI high"], tab.loc[k, "P>0"] = tab.loc[k, "MACRO"] - tab.iloc[0]["MACRO"], np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()
print(tab.round(3).to_string())
sp = lambda a, b: np.mean([pd.Series(a[:, j]).corr(pd.Series(b[:, j]), method="spearman") for j in range(len(T))])
print(f"\nmean Spearman on gold-58: A~B {sp(A, B):.3f}, A~R {sp(A, R):.3f}, B~R {sp(B, R):.3f}")
