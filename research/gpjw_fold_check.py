"""Validation of one training session of the public 2.5D reader recipe (kaggle/rsna-knee-gpjw/).

For the fold models named on the command line (all trained on the same fold):
  - gold-58 against the public fold checkpoints (did our training reach their level?)
  - held-out fold AUC against report labels, with paired bootstrap between models and for their rank-mean
  - per-finding table, and the gold-58 AUC of the label table itself (is the model closer to the
    radiologists than the labels it was trained on? the condition for mixing its predictions into them)

Usage (from the RSNA project folder):
  python research/gpjw_fold_check.py runs/rsna-knee-gpjw-train-r1/latest/res ctrl cnv2 --fold 3
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_BOOT = 2000


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def macro(y, p, ok=None, rows=None):
    if rows is not None:
        y, p, ok = y[rows], p[rows], (None if ok is None else ok[rows])
    out = [auc(y[:, j] if ok is None else y[ok[:, j], j], p[:, j] if ok is None else p[ok[:, j], j]) for j in range(y.shape[1])]
    return float(np.nanmean(out)), out


def boot(y, a, b, ok=None, seed=0):
    rng = np.random.default_rng(seed)
    d = np.array([macro(y, b, ok, r)[0] - macro(y, a, ok, r)[0] for r in (rng.integers(0, len(y), len(y)) for _ in range(N_BOOT))])
    return np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()


ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("tags", nargs="+"); ap.add_argument("--fold", type=int, required=True)
a = ap.parse_args()
lab = ld("runs/solo/gpjw_labels_v0.csv")
gold_ids = lab.index[lab.is_gold == 1]
gy = lab.loc[gold_ids, [f"gold_{t}" for t in T]].values.astype(int)
oof = {t: ld(Path(a.res) / f"oof_{t}_fold{a.fold}.csv") for t in a.tags}
val_ids = oof[a.tags[0]].index[oof[a.tags[0]].set == "val"]
assert all((o.index[o.set == "val"] == val_ids).all() for o in oof.values())
rk = lambda d: d.rank(pct=True)

# ---- gold-58 ----
print(f"== gold-58 (n = {len(gold_ids)}), macro AUC ==")
g = {t: oof[t].loc[gold_ids, T] for t in a.tags}
pub = {f"public f{k}": ld(f"runs/rsna-knee-gpjw-check/latest/gpjw_cnxt_v0_fold{k}.csv").loc[gold_ids, T] for k in (0, 1, 2)}
solo_files = sorted(Path("runs").glob("rsna-knee-train-solo-*/latest/res/gold_cnx_224_pilk_one_f*_s0.csv"))
rows = {**pub, **{f"ours {t} f{a.fold}": g[t] for t in a.tags}}
rows["public 3 folds (mean)"] = sum(pub.values()) / 3
rows[f"public 3 + ours {a.tags[0]}"] = (sum(pub.values()) + g[a.tags[0]]) / 4
if len(a.tags) > 1:
    rows["public 3 + ours " + " + ".join(a.tags)] = (sum(pub.values()) + sum(g.values())) / (3 + len(a.tags))
    rows["ours " + " + ".join(a.tags) + " (rank mean)"] = sum(rk(v) for v in g.values()) / len(a.tags)
rows["consensus labels (trained on)"] = lab.loc[gold_ids, T]
if solo_files:
    rows["old solo model, 5 folds"] = sum(rk(ld(f).loc[gold_ids, T]) for f in solo_files) / len(solo_files)
tab = pd.DataFrame({k: macro(gy, v.values)[1] for k, v in rows.items()}, index=T).T
tab.insert(0, "MACRO", tab.mean(axis=1))
print(tab.round(3).to_string())
ref = sum(pub.values()) / 3
for k in (f"public 3 + ours {a.tags[0]}",) + (("public 3 + ours " + " + ".join(a.tags),) if len(a.tags) > 1 else ()):
    lo, hi, p = boot(gy, ref.values, rows[k].values)
    print(f"{k} vs public 3 folds: {tab.loc[k, 'MACRO'] - tab.loc['public 3 folds (mean)', 'MACRO']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
lo, hi, p = boot(gy, lab.loc[gold_ids, T].values, g[a.tags[0]].values)
print(f"ours {a.tags[0]} vs the labels it was trained on: {tab.loc[f'ours {a.tags[0]} f{a.fold}', 'MACRO'] - tab.loc['consensus labels (trained on)', 'MACRO']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")

# ---- held-out fold ----
vy = (lab.loc[val_ids, T].values > 0.5).astype(int)
sl4 = pd.read_csv("research/public_notebooks/knee_g2/lab2/llm_labels_v4_blend.csv", dtype={"StudyInstanceUID": str}).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T].reindex(val_ids)
sy, sok = (sl4.values > 0.5).astype(int), (2 * sl4 - 1).abs().values >= 0.5
v = {t: oof[t].loc[val_ids, T] for t in a.tags}
cand = {f"ours {t}": v[t] for t in a.tags}
if len(a.tags) > 1:
    cand["rank mean of " + " + ".join(a.tags)] = sum(rk(x) for x in v.values()) / len(a.tags)
solo = ld("runs/solo/oof_round1.csv").reindex(val_ids)[T]
if solo.notna().all().all():
    cand["old solo model (its out-of-fold)"] = solo
print(f"\n== fold {a.fold}, held out (n = {len(val_ids)}): macro AUC vs report labels ==")
base = cand[f"ours {a.tags[0]}"].values
for k, x in cand.items():
    m1, m2 = macro(vy, x.values)[0], macro(sy, x.values, sok)[0]
    line = f"{k:34s} consensus>0.5: {m1:.4f} | sl4 confident cells: {m2:.4f}"
    if k != f"ours {a.tags[0]}":
        lo, hi, p = boot(vy, base, x.values)
        line += f" | vs {a.tags[0]}: {m1 - macro(vy, base)[0]:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], P(better) {p:.2f}"
    print(line)
per = pd.DataFrame({k: macro(vy, x.values)[1] for k, x in cand.items()}, index=T)
print(per.round(3).to_string())
if len(a.tags) > 1:
    c = [v[a.tags[0]][t].corr(v[a.tags[1]][t], method="spearman") for t in T]
    print(f"Spearman {a.tags[0]} ~ {a.tags[1]} on the held-out fold: {np.mean(c):.3f} (min {np.min(c):.3f} {T[int(np.argmin(c))]})")
