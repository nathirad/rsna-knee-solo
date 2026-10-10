"""Out-of-fold tools for the solo member (see kaggle/rsna-knee-train-solo/).

  merge  join the per-fold oof_*.csv files of one training round into one table
  score  macro AUC of an out-of-fold table against the report labels (and gold-58)
  diff   paired bootstrap of two out-of-fold tables: is B better than A, or is it noise?

The merged table is also the pseudo-label input of the next round (build.py --pseudo).
The reference is the same one the training job scores its holdout on: stevenleehans v4
blend, binarised, confident cells only. It is never a table that holds pseudo labels.

Usage (from the RSNA project folder):
  python research/oof_tools.py merge runs/rsna-knee-train-solo-*/latest/res --out runs/solo/oof_round1.csv
  python research/oof_tools.py score runs/solo/oof_round1.csv [--train-csv data/train.csv --gold-dirs ...]
  python research/oof_tools.py diff  runs/solo/oof_round1.csv runs/solo/oof_round2.csv
"""
import argparse, glob, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
REF = Path(__file__).parent / "public_notebooks/knee_g2/lab2/llm_labels_v4_blend.csv"
MIN_GAIN = 0.003      # below this an out-of-fold difference is treated as noise (PLAN)


def reference(index, ref_csv=REF):
    ref = pd.read_csv(ref_csv).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[T]
    ref = ref.reindex(index)
    return (ref.values > 0.5).astype(int), ((2 * ref - 1).abs().values >= 0.5)


def macro(y, ok, p, rows=None):
    out = []
    for j in range(len(T)):
        m = ok[:, j] if rows is None else ok[rows, j]
        yy = (y[:, j] if rows is None else y[rows, j])[m]
        pp = (p[:, j] if rows is None else p[rows, j])[m]
        out.append(roc_auc_score(yy, pp) if len(set(yy)) > 1 else np.nan)
    return float(np.nanmean(out)), out


def load(path):
    d = pd.read_csv(path, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    assert d.index.is_unique, f"{path}: a study appears twice"
    return d


def cmd_merge(a):
    files = sorted(f for d in a.dirs for f in glob.glob(os.path.join(d, "oof_*.csv")))
    if a.tag:
        files = [f for f in files if a.tag in os.path.basename(f)]
    if not files:
        raise SystemExit("no oof_*.csv found")
    parts = [pd.read_csv(f, dtype={"StudyInstanceUID": str}) for f in files]
    tags = {os.path.basename(f).rsplit("_f", 1)[0] for f in files}
    if len(tags) != 1:
        raise SystemExit(f"more than one configuration in these folders, pick one with --tag: {sorted(tags)}")
    oof = pd.concat(parts, ignore_index=True)
    dup = oof["StudyInstanceUID"].duplicated().sum()
    if dup:
        raise SystemExit(f"{dup} studies appear in more than one fold file (same fold merged twice?)")
    folds = sorted(oof["fold"].unique().tolist())
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(a.out, index=False)
    print(f"{tags.pop()}: {len(files)} file(s), folds {folds}, {len(oof)} studies -> {a.out}")
    if folds != [0, 1, 2, 3, 4]:
        print("WARNING: not all five folds are present; this table is not yet a full out-of-fold table")


def cmd_score(a):
    oof = load(a.oof)
    y, ok = reference(oof.index)
    auc, per = macro(y, ok, oof[T].values)
    print(f"out-of-fold macro AUC vs report labels: {auc:.4f}  "
          f"(n = {len(oof)}, folds {sorted(int(f) for f in oof['fold'].unique())})")
    for f in sorted(oof["fold"].unique()):
        r = np.where(oof["fold"].values == f)[0]
        print(f"  fold {f}: {macro(y, ok, oof[T].values, r)[0]:.4f}  (n = {len(r)})")
    print(pd.Series(per, index=T).round(4).to_string())
    if a.train_csv and a.gold_dirs:
        tr = pd.read_csv(a.train_csv, dtype={"StudyInstanceUID": str})
        g = tr.dropna(subset=T).set_index("StudyInstanceUID")[T].astype(int)
        files = sorted(f for d in a.gold_dirs for f in glob.glob(os.path.join(d, "gold_*.csv"))
                       if a.tag in os.path.basename(f))
        preds = [load(f).loc[g.index, T].rank(pct=True) for f in files]
        p = sum(preds) / len(preds)
        gold = np.mean([roc_auc_score(g[t], p[t]) for t in T])
        print(f"gold-58 (rank-mean of {len(files)} fold model(s)): {gold:.4f}   [sanity check only, n = {len(g)}]")


def cmd_diff(a):
    A, B = load(a.a), load(a.b)
    idx = A.index.intersection(B.index)
    if len(idx) < 0.95 * max(len(A), len(B)):
        print(f"WARNING: the tables share only {len(idx)} studies")
    y, ok = reference(idx)
    pa, pb = A.loc[idx, T].values, B.loc[idx, T].values
    a0, b0 = macro(y, ok, pa)[0], macro(y, ok, pb)[0]
    rng, d = np.random.default_rng(0), []
    for _ in range(a.n_boot):
        r = rng.integers(0, len(idx), len(idx))
        d.append(macro(y, ok, pb, r)[0] - macro(y, ok, pa, r)[0])
    d = np.array(d)
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"A {a0:.4f} | B {b0:.4f} | B - A = {b0 - a0:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}], P(B > A) = {(d > 0).mean():.2f}")
    verdict = ("B is better" if b0 - a0 >= MIN_GAIN and lo > 0 else
               "A is better" if a0 - b0 >= MIN_GAIN and hi < 0 else "no real difference (treat as noise)")
    print(f"rule (gain >= {MIN_GAIN} and the CI excludes 0): {verdict}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("merge"); m.add_argument("dirs", nargs="+"); m.add_argument("--out", required=True)
    m.add_argument("--tag", default="", help="keep only files whose name contains this")
    s = sub.add_parser("score"); s.add_argument("oof"); s.add_argument("--train-csv")
    s.add_argument("--gold-dirs", nargs="*")
    s.add_argument("--tag", default="", help="use only gold files whose name contains this")
    d = sub.add_parser("diff"); d.add_argument("a"); d.add_argument("b"); d.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    {"merge": cmd_merge, "score": cmd_score, "diff": cmd_diff}[a.cmd](a)
