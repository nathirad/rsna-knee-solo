"""Validation of the label-policy fine-tunes (tag polA) of the 2.5D reader.

polA = the fold model of the same fold, trained 4 more epochs with Effusion and Synovitis targets taken
from pilkwang's graded scores (loss weight 0.25 + 0.75*confidence); the other ten findings unchanged.
Folds 0, 1 start from the public checkpoints, folds 3, 4 from our `ctrl` checkpoints; fold 2 is still
the public checkpoint.

Printed:
  1. gold-58 per fold, before -> after, for Effusion, Synovitis, the other ten findings and the macro
  2. gold-58 of the five models together (probability mean, as the submission notebook's reader does),
     before vs after, with a paired bootstrap; and against the old solo model on the two findings
  3. the two rules written down before the results were read
  4. held-out folds against report labels (the ten unchanged findings must not get worse)
  5. the stack + reader mix at weight 0.30 on gold-58, before vs after (the level is biased toward the
     stack, the difference between two reader legs is not)

Usage (from the RSNA project folder): python research/gpjw_polA_check.py
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
FIX = ["Effusion", "Synovitis"]
REST = [t for t in T if t not in FIX]
R1, R2 = Path("runs/rsna-knee-gpjw-train-r1/latest/res"), Path("runs/rsna-knee-gpjw-train-r2/latest/res")
R4, R5 = Path("runs/rsna-knee-gpjw-train-r4/latest/res"), Path("runs/rsna-knee-gpjw-train-r5/latest/res")
OLD = {0: R2 / "oof_public_fold0.csv", 1: R2 / "oof_public_fold1.csv", 2: R2 / "oof_public_fold2.csv",
       3: R1 / "oof_ctrl_fold3.csv", 4: R2 / "oof_ctrl_fold4.csv"}
NEW = {0: R5 / "oof_polA_fold0.csv", 1: R5 / "oof_polA_fold1.csv", 2: OLD[2],
       3: R4 / "oof_polA_fold3.csv", 4: R4 / "oof_polA_fold4.csv"}
TUNED = [0, 1, 3, 4]
N_BOOT = 4000
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(y, p, rows=None, ok=None):
    if rows is not None:
        y, p, ok = y[rows], p[rows], (None if ok is None else ok[rows])
    return np.array([auc(y[:, j] if ok is None else y[ok[:, j], j], p[:, j] if ok is None else p[ok[:, j], j]) for j in range(y.shape[1])])


def ci(d):
    d = d[~np.isnan(d)]
    return f"95% CI [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}], P(better) {(d > 0).mean():.2f}"


rk = lambda a: np.apply_along_axis(rankdata, 0, a) / len(a)
jx = lambda names: [T.index(t) for t in names]
lab = ld("runs/solo/gpjw_labels_v0.csv")
gid = lab.index[lab.is_gold == 1]
y = lab.loc[gid, [f"gold_{t}" for t in T]].values.astype(int)
old = {k: ld(p) for k, p in OLD.items()}
new = {k: ld(p) for k, p in NEW.items()}
g_old = {k: d[d.set == "gold"].loc[gid, T].values for k, d in old.items()}
g_new = {k: d[d.set == "gold"].loc[gid, T].values for k, d in new.items()}
rng = np.random.default_rng(0)
boots = [rng.integers(0, len(y), len(y)) for _ in range(N_BOOT)]

# ---- 1. per fold ----
print(f"== 1. gold-58 (n = {len(gid)}; positives: " + ", ".join(f"{t} {int(y[:, T.index(t)].sum())}" for t in FIX) + ") per fold, before -> after ==")
rows = []
for k in TUNED:
    a, b = per(y, g_old[k]), per(y, g_new[k])
    rows.append({"fold": k, **{f"{t} before": a[T.index(t)] for t in FIX}, **{f"{t} after": b[T.index(t)] for t in FIX},
                 "other 10 before": a[jx(REST)].mean(), "other 10 after": b[jx(REST)].mean(), "macro before": a.mean(), "macro after": b.mean()})
tab = pd.DataFrame(rows).set_index("fold")
tab = tab[["Effusion before", "Effusion after", "Synovitis before", "Synovitis after", "other 10 before", "other 10 after", "macro before", "macro after"]]
print(pd.concat([tab, tab.mean().to_frame("mean").T]).round(4).to_string())
d_eff = float((tab["Effusion after"] - tab["Effusion before"]).mean())
d_syn = float((tab["Synovitis after"] - tab["Synovitis before"]).mean())
for name, j in (("Effusion", T.index("Effusion")), ("Synovitis", T.index("Synovitis")), ("macro", None)):
    f = (lambda p, b: auc(y[b, j], p[b, j])) if j is not None else (lambda p, b: np.nanmean(per(y, p, b)))
    d = np.array([np.mean([f(g_new[k], b) - f(g_old[k], b) for k in TUNED]) for b in boots[:2000]])
    print(f"  mean over the 4 folds, after - before, {name:9s}: {np.nanmean([f(g_new[k], np.arange(len(y))) - f(g_old[k], np.arange(len(y))) for k in TUNED]):+.4f}, {ci(d)}")

# ---- 2. five models together ----
E_old, E_new = sum(g_old.values()) / 5, sum(g_new.values()) / 5
solo = sum(rk(ld(f).loc[gid, T].values) for f in sorted(glob.glob("runs/rsna-knee-train-solo-*/latest/res/gold_cnx_224_pilk_one_f*_s0.csv"))) / 5
routed = rk(E_old).copy()
routed[:, jx(FIX)] = solo[:, jx(FIX)]
t2 = pd.DataFrame({"reader before (5)": per(y, E_old), "reader after (5)": per(y, E_new), "old solo (5)": per(y, solo),
                   "labels: consensus": per(y, lab.loc[gid, T].values)}, index=T)
t2["after - before"] = t2["reader after (5)"] - t2["reader before (5)"]
print("\n== 2. gold-58, the five models together ==")
print(pd.concat([t2, t2.mean().to_frame("MACRO").T]).round(3).to_string())
m_old, m_new = per(y, E_old).mean(), per(y, E_new).mean()
d = np.array([np.nanmean(per(y, E_new, b)) - np.nanmean(per(y, E_old, b)) for b in boots])
print(f"macro: before {m_old:.4f}, after {m_new:.4f} ({m_new - m_old:+.4f}), {ci(d)}")
for t in FIX:
    j = T.index(t)
    d = np.array([auc(y[b, j], E_new[b, j]) - auc(y[b, j], E_old[b, j]) for b in boots])
    print(f"{t}: before {auc(y[:, j], E_old[:, j]):.4f}, after {auc(y[:, j], E_new[:, j]):.4f}, {ci(d)}; old solo {auc(y[:, j], solo[:, j]):.4f}")
print(f"for reference, reader before with both findings taken from the old solo model: macro {per(y, routed).mean():.4f}")
sp = lambda a, b: float(pd.Series(a).corr(pd.Series(b), method="spearman"))
for t in FIX:
    j = T.index(t)
    print(f"Spearman on gold, {t}: before~solo {sp(E_old[:, j], solo[:, j]):.3f}, after~solo {sp(E_new[:, j], solo[:, j]):.3f}, before~after {sp(E_old[:, j], E_new[:, j]):.3f}")

# ---- 3. the rules ----
print("\n== 3. rules written before the results were read ==")
ok_a, ok_b = d_eff >= 0.04, m_new >= 0.913 + 0.004
print(f"(a) Effusion on gold-58 rises by at least 0.04 on average over the tuned folds: {d_eff:+.4f} -> {'MET' if ok_a else 'NOT MET'}")
print(f"(b) macro of the five models together is at least 0.917 (0.913 + 0.004): {m_new:.4f} -> {'MET' if ok_b else 'NOT MET'}")
print("both met: submit one" if ok_a and ok_b else "not both met: no submission on this evidence")

# ---- 4. held-out folds against report labels ----
print("\n== 4. held-out fold, AUC against report labels (consensus > 0.5), before -> after ==")
polA = ld("runs/solo/gpjw_labels_polA.csv")
rows = []
for k in TUNED:
    vo, vn = old[k][old[k].set == "val"], new[k][new[k].set == "val"]
    assert (vo.index == vn.index).all() and (lab.loc[vo.index, "fold"] == k).all()
    cy = (lab.loc[vo.index, T].values > 0.5).astype(int)
    a, b = per(cy, vo[T].values), per(cy, vn[T].values)
    py = (polA.loc[vo.index, FIX].values > 0.5).astype(int)
    pa, pb = per(py, vo[FIX].values), per(py, vn[FIX].values)
    rows.append({"fold": k, "other 10 before": a[jx(REST)].mean(), "other 10 after": b[jx(REST)].mean(),
                 "Effusion before": a[T.index("Effusion")], "Effusion after": b[T.index("Effusion")],
                 "Synovitis before": a[T.index("Synovitis")], "Synovitis after": b[T.index("Synovitis")],
                 "Eff vs graded before": pa[0], "Eff vs graded after": pb[0], "Syn vs graded before": pa[1], "Syn vs graded after": pb[1]})
t4 = pd.DataFrame(rows).set_index("fold")
print(pd.concat([t4, t4.mean().to_frame("mean").T]).round(4).to_string())
print("(Effusion and Synovitis against the consensus labels are expected to move: the targets changed, and the consensus is the label this fix distrusts)")

# ---- 5. in the stack mix ----
stack = ld("runs/rsna-knee-base-gold/latest/submission.csv").loc[gid, T].values
mix = lambda leg, w: rk((1 - w) * rk(stack) + w * rk(leg))
print("\n== 5. gold-58, stack + reader at weight w (the stack was fitted to these studies: read the difference, not the level) ==")
print(f"stack alone {per(y, stack).mean():.4f}")
for w in (0.2, 0.3, 0.4):
    a, b = per(y, mix(E_old, w)), per(y, mix(E_new, w))
    d = np.array([np.nanmean(per(y, mix(E_new, w), r)) - np.nanmean(per(y, mix(E_old, w), r)) for r in boots[:2000]])
    print(f"w={w}: reader before {a.mean():.4f}, reader after {b.mean():.4f} ({b.mean() - a.mean():+.4f}), {ci(d)}"
          f" | Effusion {a[T.index('Effusion')]:.3f} -> {b[T.index('Effusion')]:.3f}, Synovitis {a[T.index('Synovitis')]:.3f} -> {b[T.index('Synovitis')]:.3f}")
