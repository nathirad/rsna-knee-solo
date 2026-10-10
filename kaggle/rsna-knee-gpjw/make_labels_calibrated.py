"""Label table whose targets mean what the experts mean.

A report label and an expert label are different things: on the 58 expert-labelled studies a hedged
report finding is expert-positive far less often than the consensus target says (ACL, MCL: about 0 of
the hedged ones; Effusion: half). This table replaces every target by P(expert positive | report labels),
with the mapping of research/label_calibration.py fitted on the 58 gold studies:
  --kind finding   one logistic curve per finding on logit(consensus), shrunk toward the all-findings curve
  --kind pooled    one curve for all findings
  --kind grades    per finding, the expert-positive share of each pilkwang grade level (shrunk, monotone)
  --kind a+b       the mean of two mappings' targets, e.g. finding+grades: the policy research/head_lab.py found best
                   (gold-58 of two retrained fold heads 0.909 -> 0.918; consensus alone for a+b = raw+grades)
  --weights conf   loss weight 0.3 + 0.7*|2*target - 1| (cells the mapping is unsure about count less)
  orig_<f> = the consensus label, used only to score the held-out fold
Gold rows are never trained on and keep the consensus.

Usage (from the RSNA project folder): python kaggle/rsna-knee-gpjw/make_labels_calibrated.py --kind finding --name calF
"""
import argparse, sys
from pathlib import Path

import numpy as np

RSNA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RSNA / "research"))
ap = argparse.ArgumentParser()
ap.add_argument("--kind", required=True, help="finding | pooled | grades | raw, or several joined by +")
ap.add_argument("--weights", default="one", choices=["one", "conf"])
ap.add_argument("--name", required=True)
a = ap.parse_args()
import os
os.chdir(RSNA)                                           # label_calibration.py reads its tables relative to the project folder
from label_calibration import Calibrator, T, load       # noqa: E402

lab, cons, lev, src = load()
gold = (lab.is_gold == 1).values
y = lab.loc[gold, [f"gold_{c}" for c in T]].values.astype(float)
kinds = a.kind.split("+")
assert set(kinds) <= {"finding", "pooled", "grades", "raw"}, kinds
target = np.mean([Calibrator(k).fit(cons[gold], lev[gold], src[gold], y).predict(cons, lev, src) for k in kinds], 0).clip(0.01, 0.99)
out = lab.copy()
for j, t in enumerate(T):
    out[f"orig_{t}"] = lab[t]
    out[f"w_{t}"] = 0.3 + 0.7 * np.abs(2 * target[:, j] - 1) if a.weights == "conf" else 1.0
    out.loc[~gold, t] = target[~gold, j]
    out.loc[gold, f"w_{t}"] = 1.0
path = RSNA / f"runs/solo/gpjw_labels_{a.name}.csv"
out.to_csv(path, float_format="%.6g")
print(f"saved {path} {out.shape}: mapping {a.kind}, weights {a.weights}")
print("mean target on the report-labelled studies:", dict(zip(T, target[~gold].mean(0).round(2))))
print("mean consensus there:                     ", dict(zip(T, cons[~gold].mean(0).round(2))))
