"""Label table of a mixed-label round: target = a * out-of-fold prediction + (1 - a) * label.

The out-of-fold table (research/gpjw_oof_table.py) holds, for every non-gold study, the prediction
of a fold model that did not train on it. The original labels are kept in `orig_<finding>` columns:
the training job scores its held-out fold against those, so rounds stay comparable. Folds are unchanged.

Note on the held-out fold score of a mixed round: the predictions mixed into the other folds'
targets come from models that saw this fold's report labels, so that score is a little optimistic.
Gold-58 is never trained on in any round and stays the clean check.

Usage (from the RSNA project folder): python kaggle/rsna-knee-gpjw/make_labels_mixed.py --alpha 0.5
"""
import argparse
from pathlib import Path

import pandas as pd

RSNA = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--labels", default=str(RSNA / "runs/solo/gpjw_labels_v0.csv"))
ap.add_argument("--oof", default=str(RSNA / "runs/solo/gpjw_oof_round1.csv"))
ap.add_argument("--out", default="")
a = ap.parse_args()
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
lab, oof = ld(a.labels), ld(a.oof)
weak = lab.index[lab.is_gold == 0]
assert sorted(oof.index) == sorted(weak), "the out-of-fold table must cover every non-gold study exactly once"
assert (oof.loc[weak, "fold"] == lab.loc[weak, "fold"]).all(), "fold assignment differs between the two tables"
assert oof[T].notna().all().all() and ((oof[T] >= 0) & (oof[T] <= 1)).all().all()
out = lab.copy()
for t in T:
    out[f"orig_{t}"] = lab[t]
out.loc[weak, T] = (a.alpha * oof.loc[weak, T] + (1 - a.alpha) * lab.loc[weak, T]).clip(0.01, 0.99)
path = Path(a.out or RSNA / f"runs/solo/gpjw_labels_mix{int(round(a.alpha * 100)):02d}.csv")
out.to_csv(path, float_format="%.6g")      # short numbers: the table is embedded in the session notebook
moved = (out.loc[weak, T] - lab.loc[weak, T]).abs()
flip = ((out.loc[weak, T] > 0.5) != (lab.loc[weak, T] > 0.5)).mean()
print(f"saved {path} {out.shape}: alpha {a.alpha}, mean |change| {moved.values.mean():.3f}, cells crossing 0.5: {flip.mean():.3%}")
print("cells crossing 0.5 per finding:", flip.round(3).to_dict())
