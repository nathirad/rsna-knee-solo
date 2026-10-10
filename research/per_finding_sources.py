"""Which of our own models is best for which finding, on the 58 expert-labelled studies?

The 2.5D reader (public recipe, consensus labels, plain BCE) and our older solo model (pilkwang labels,
loss weight 0.25 + 0.75*confidence) were trained with different label handling. This compares them
per finding, tests a per-finding choice between them, and estimates how much of that choice's gain
survives when the choice is made on one half of the gold studies and scored on the other half.

Usage (from the RSNA project folder): python research/per_finding_sources.py
"""
import glob

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
o3, o4 = ld("runs/rsna-knee-gpjw-train-r1/latest/res/oof_ctrl_fold3.csv"), ld("runs/rsna-knee-gpjw-train-r2/latest/res/oof_ctrl_fold4.csv")
reader = (sum(ld(f"runs/rsna-knee-gpjw-check/latest/gpjw_cnxt_v0_fold{k}.csv").loc[gid, T].values for k in (0, 1, 2))
          + o3[o3.set == "gold"].loc[gid, T].values + o4[o4.set == "gold"].loc[gid, T].values) / 5
solo = sum(rk(ld(f).loc[gid, T].values) for f in sorted(glob.glob("runs/rsna-knee-train-solo-*/latest/res/gold_cnx_224_pilk_one_f*_s0.csv"))) / 5
stack = ld("runs/rsna-knee-base-gold/latest/submission.csv").loc[gid, T].values
m448 = ld("runs/rsna-knee-m448-check/latest/m448_train_check.csv").loc[gid, T].values
R, S = rk(reader), rk(solo)
tab = pd.DataFrame({"reader (5 folds)": per(y, R), "old solo (5 folds)": per(y, S), "rank mean of both": per(y, (R + S) / 2),
                    "M448": per(y, m448), "stack (fitted to gold)": per(y, stack),
                    "labels: consensus": per(y, lab.loc[gid, T].values)}, index=T)
tab["solo - reader"] = tab["old solo (5 folds)"] - tab["reader (5 folds)"]
print(pd.concat([tab, tab.mean().to_frame("MACRO").T]).round(3).to_string())

rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(4000)]
print("\nper finding, solo - reader on gold-58 with a paired bootstrap:")
for j, t in enumerate(T):
    d = np.array([auc(y[b, j], S[b, j]) - auc(y[b, j], R[b, j]) for b in boots])
    d = d[~np.isnan(d)]
    flag = "  <-- solo better" if np.percentile(d, 2.5) > 0 else ("  <-- reader better" if np.percentile(d, 97.5) < 0 else "")
    print(f"  {t:17s} {tab.loc[t, 'solo - reader']:+.3f}  95% CI [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]{flag}")


def route(choice):
    out = R.copy()
    for j in choice:
        out[:, j] = S[:, j]
    return out


for name, cols in (("Effusion from solo", ["Effusion"]), ("Effusion + Synovitis from solo", ["Effusion", "Synovitis"])):
    H = route([T.index(c) for c in cols])
    d = np.array([np.nanmean(per(y, H, b)) - np.nanmean(per(y, R, b)) for b in boots])
    print(f"\nreader with {name}: macro {per(y, H).mean():.4f} vs reader {per(y, R).mean():.4f} "
          f"({per(y, H).mean() - per(y, R).mean():+.4f}, 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}])")

# honest estimate of a per-finding choice: choose on one half, score on the other half
gains, picks = [], np.zeros(len(T))
for _ in range(2000):
    perm = rng.permutation(len(y)); a, b = perm[:29], perm[29:]
    sel = [j for j in range(len(T)) if auc(y[a, j], S[a, j]) > auc(y[a, j], R[a, j]) + 0.03]
    picks[sel] += 1
    g = np.nanmean(per(y, route(sel), b)) - np.nanmean(per(y, R, b))
    gains.append(g)
gains = np.array(gains)
print(f"\nchoice made on half of gold (solo if > reader + 0.03), scored on the other half, 2000 splits: "
      f"mean gain {gains.mean():+.4f}, 5th-95th percentile [{np.percentile(gains, 5):+.4f}, {np.percentile(gains, 95):+.4f}], P(gain > 0) {(gains > 0).mean():.2f}")
print("how often each finding was routed to solo:", {t: round(p / 2000, 2) for t, p in zip(T, picks) if p > 100})

# does it carry into the mix with the stack? (absolute level is biased toward the stack; compare the two legs at the same weight)
mix = lambda leg, w: rk((1 - w) * rk(stack) + w * leg)
H = route([T.index("Effusion"), T.index("Synovitis")])
for w in (0.3,):
    d = np.array([np.nanmean(per(y, mix(H, w), b)) - np.nanmean(per(y, mix(R, w), b)) for b in boots])
    print(f"\nstack + leg at w={w}: reader leg {per(y, mix(R, w)).mean():.4f}, reader with Effusion+Synovitis from solo {per(y, mix(H, w)).mean():.4f} "
          f"({per(y, mix(H, w)).mean() - per(y, mix(R, w)).mean():+.4f}, 95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}])")
