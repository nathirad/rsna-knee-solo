"""Label table with a per-finding label policy.

For most findings the reader trained on the four-table consensus beats our older model. For Effusion
(and less clearly Synovitis) the older model is far better on the expert-labelled studies (Effusion
0.970 vs 0.877, research/per_finding_sources.py). It was trained on pilkwang's graded scores with a
loss weight of 0.25 + 0.75*confidence, so cells the report is silent on (verdict UNK) count little
and the image decides. This table gives the named findings that treatment and leaves the others on
the consensus:
  target   = pilkwang graded score for the named findings, consensus otherwise
  w_<f>    = 0.25 + 0.75 * pilkwang confidence for the named findings, 1 otherwise
  orig_<f> = the consensus label, used only to score the held-out fold

Usage (from the RSNA project folder): python kaggle/rsna-knee-gpjw/make_labels_policy.py --findings Effusion Synovitis --name polA
"""
import argparse
from pathlib import Path

import pandas as pd

RSNA = Path(__file__).resolve().parents[2]
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ap = argparse.ArgumentParser()
ap.add_argument("--findings", nargs="+", required=True)
ap.add_argument("--name", required=True)
a = ap.parse_args()
assert set(a.findings) <= set(T)
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
lab = ld(RSNA / "runs/solo/gpjw_labels_v0.csv")
pilk = ld(RSNA / "research/public_notebooks/knee_g2/lab1/report_labels_v2.csv").reindex(lab.index)
out = lab.copy()
for t in T:
    out[f"orig_{t}"] = lab[t]
    out[f"w_{t}"] = 1.0
for t in a.findings:
    has = pilk[t].notna()
    out.loc[has, t] = pilk.loc[has, t].clip(0.01, 0.99)
    out.loc[has, f"w_{t}"] = 0.25 + 0.75 * pilk.loc[has, f"{t}__conf"]
gold = lab.is_gold == 1
out.loc[gold, T] = lab.loc[gold, T]                    # gold rows are never trained on; left as they were
path = RSNA / f"runs/solo/gpjw_labels_{a.name}.csv"
out.to_csv(path, float_format="%.6g")
weak = ~gold
print(f"saved {path} {out.shape}; findings on the graded-score policy: {a.findings}")
for t in a.findings:
    print(f"  {t}: verdicts {pilk.loc[weak, t + '__verdict'].value_counts().to_dict()}, mean target {out.loc[weak, t].mean():.3f} "
          f"(consensus {lab.loc[weak, t].mean():.3f}), mean weight {out.loc[weak, 'w_' + t].mean():.3f}")
