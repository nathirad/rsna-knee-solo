"""Validation of the retrained heads (kaggle/rsna-knee-gpjw/fit_heads.py) of the five reader fold models.

A head session keeps every fold model's backbone and retrains its head on another label table
(calFG: targets = P(expert positive), the mean of the calibrated consensus and the calibrated pilkwang
grade; research/label_calibration.py, research/head_lab.py). Printed, on the 58 expert-labelled studies:
  1. per fold, old head -> new head
  2. the five fold models together, with a paired bootstrap, per finding
  3. the stack + reader mix at weight 0.30, old heads vs new heads (the level is biased toward the
     stack, the difference between two reader legs is not), also with Effusion and Synovitis taken
     from the older solo model as in rsna-knee-stack-route
  4. the two rules written down before the five-fold result was read

Usage (from the RSNA project folder): python research/gpjw_heads_check.py [--name calFG] [--run rsna-knee-gpjw-heads-a]
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
R1, R2 = Path("runs/rsna-knee-gpjw-train-r1/latest/res"), Path("runs/rsna-knee-gpjw-train-r2/latest/res")
OLD = {0: R2 / "oof_public_fold0.csv", 1: R2 / "oof_public_fold1.csv", 2: R2 / "oof_public_fold2.csv",
       3: R1 / "oof_ctrl_fold3.csv", 4: R2 / "oof_ctrl_fold4.csv"}
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
ap = argparse.ArgumentParser()
ap.add_argument("--name", default="calFG")
ap.add_argument("--run", default="rsna-knee-gpjw-heads-a")
a = ap.parse_args()
NEW = {k: Path(f"runs/{a.run}/latest/res/oof_{a.name}_fold{k}.csv") for k in range(5)}


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(y, p, rows=None):
    if rows is not None:
        y, p = y[rows], p[rows]
    return np.array([auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


def ci(d):
    return f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], P(better) {(d > 0).mean():.2f}"


rk = lambda x: np.apply_along_axis(rankdata, 0, x) / len(x)
lab = ld("runs/solo/gpjw_labels_v0.csv")
gid = lab.index[lab.is_gold == 1]
y = lab.loc[gid, [f"gold_{t}" for t in T]].values.astype(int)
g = lambda p: (lambda d: d[d.set == "gold"].loc[gid, T].values)(ld(p))
old, new = {k: g(p) for k, p in OLD.items()}, {k: g(p) for k, p in NEW.items()}
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(4000)]

print(f"== 1. gold-58 per fold, old head -> new head ({a.name}) ==")
t1 = pd.DataFrame({"old head": [per(y, old[k]).mean() for k in range(5)], "new head": [per(y, new[k]).mean() for k in range(5)]})
t1["change"] = t1["new head"] - t1["old head"]
print(pd.concat([t1, t1.mean().to_frame("mean").T]).round(4).to_string())

E_old, E_new = sum(old.values()) / 5, sum(new.values()) / 5
t2 = pd.DataFrame({"old heads": per(y, E_old), "new heads": per(y, E_new)}, index=T)
t2["change"] = t2["new heads"] - t2["old heads"]
lo, hi = [], []
for j in range(len(T)):
    d = np.array([auc(y[b, j], E_new[b, j]) - auc(y[b, j], E_old[b, j]) for b in boots[:2000]])
    d = d[~np.isnan(d)]
    lo.append(np.percentile(d, 2.5)); hi.append(np.percentile(d, 97.5))
t2["CI low"], t2["CI high"] = lo, hi
print("\n== 2. gold-58, the five fold models together ==")
print(pd.concat([t2, t2[["old heads", "new heads", "change"]].mean().to_frame("MACRO").T]).round(3).to_string())
m_old, m_new = per(y, E_old).mean(), per(y, E_new).mean()
d = np.array([np.nanmean(per(y, E_new, b)) - np.nanmean(per(y, E_old, b)) for b in boots])
print(f"macro: old heads {m_old:.4f}, new heads {m_new:.4f} ({m_new - m_old:+.4f}), {ci(d)}")

stack = ld("runs/rsna-knee-base-gold/latest/submission.csv").loc[gid, T].values
solo = sum(rk(ld(f).loc[gid, T].values) for f in sorted(glob.glob("runs/rsna-knee-train-solo-*/latest/res/gold_cnx_224_pilk_one_f*_s0.csv"))) / 5
mix = lambda leg, w=0.3: rk((1 - w) * rk(stack) + w * rk(leg))


def routed(leg, cols=("Effusion", "Synovitis")):
    L = rk(leg).copy()
    L[:, [T.index(c) for c in cols]] = solo[:, [T.index(c) for c in cols]]
    return L


print("\n== 3. gold-58, stack + reader leg at weight 0.30 (read the differences, not the level) ==")
ref = mix(E_old)
print(f"old heads (the LB 0.944 submission): {per(y, ref).mean():.4f}")
in_mix = None
for name, leg in (("new heads", E_new), ("old heads, Effusion+Synovitis from the older model (ref 56988526)", routed(E_old)),
                  ("new heads, Effusion+Synovitis from the older model", routed(E_new)), ("new heads, Synovitis from the older model", routed(E_new, ("Synovitis",)))):
    m = mix(leg)
    d = np.array([np.nanmean(per(y, m, b)) - np.nanmean(per(y, ref, b)) for b in boots[:3000]])
    delta = per(y, m).mean() - per(y, ref).mean()
    in_mix = delta if in_mix is None else in_mix
    print(f"{name:68s} {per(y, m).mean():.4f} ({delta:+.4f}), {ci(d)}")
for w in (0.2, 0.4):
    print(f"weight {w}: old heads {per(y, mix(E_old, w)).mean():.4f}, new heads {per(y, mix(E_new, w)).mean():.4f} ({per(y, mix(E_new, w)).mean() - per(y, mix(E_old, w)).mean():+.4f})")

print("\n== 4. rules written before the five-fold result was read ==")
ok1, ok2 = m_new - m_old >= 0.005, in_mix >= 0.0015
print(f"(i)  five fold models together, new heads at least +0.005 over the old heads: {m_new - m_old:+.4f} -> {'MET' if ok1 else 'NOT MET'}")
print(f"(ii) in the stack mix at 0.30, at least +0.0015 over the mix with the old heads: {in_mix:+.4f} -> {'MET' if ok2 else 'NOT MET'}")
print("both met: submit one" if ok1 and ok2 else "not both met: no submission on this evidence")
