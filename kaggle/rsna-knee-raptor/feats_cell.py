import base64, gzip, io, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Feature session for the public single CoAtNet (nartaa/rsna-knee-0949-anatomical-mirror), not a submission:
# its own preprocessing and backbone on the train studies, encode() output saved for head-only experiments.
# Built by build_feats.py; the notebook's cell is used unchanged with our extract_features() appended.
W = Path("/kaggle/working")
RES = W / "res"
RES.mkdir(parents=True, exist_ok=True)
(W / "_raptor_feats.py").write_bytes(base64.b64decode("__SCRIPT_B64__"))
lab = pd.read_csv(io.BytesIO(gzip.decompress(base64.b64decode("__LABELS_B64__"))), index_col=0)
lab.index = lab.index.astype(str)
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
order = list(lab.index[lab.is_gold == 1]) + list(lab.index[lab.is_gold == 0])      # gold first: saved even if the budget ends the run
LIMIT = __LIMIT__
pd.DataFrame({"StudyInstanceUID": order[:LIMIT] if LIMIT else order}).to_csv(W / "_studies.csv", index=False)
t0 = time.time()
rc = subprocess.run([sys.executable, "-u", str(W / "_raptor_feats.py")],
                    env={**os.environ, "RAPTOR_SPLIT": "train", "RAPTOR_STUDIES": str(W / "_studies.csv"),
                         "RAPTOR_OUT": str(RES), "RAPTOR_BUDGET_H": "__BUDGET_H__"}).returncode
print("rc", rc, "minutes", round((time.time() - t0) / 60, 1), flush=True)
assert rc == 0, "the feature script failed"

# the checkpoint's own head on the saved features, against the expert labels (plain view, no mirror)
from sklearn.metrics import roc_auc_score
z = np.load(RES / "feats_raptor_train.npz")
st = [str(s) for s in z["studies"]]
gold = [s for s in lab.index[lab.is_gold == 1] if s in set(st)]
y = lab.loc[gold, [f"gold_{c}" for c in T]].values.astype(int)
p = 1 / (1 + np.exp(-z["logit"][[st.index(s) for s in gold]]))
each = [roc_auc_score(y[:, i], p[:, i]) for i in range(len(T))]
print(f"features {z['feat'].shape} {z['feat'].dtype}; gold-{len(gold)} of the checkpoint's own head, plain view: {np.mean(each):.4f}")
print("per finding: " + " ".join(f"{c[:6]}={a:.3f}" for c, a in zip(T, each)))
json.dump({"studies": len(st), "feat_shape": list(z["feat"].shape), "gold": float(np.mean(each)), "gold_each": dict(zip(T, map(float, each))),
           "minutes": (time.time() - t0) / 60}, open(RES / "summary_raptor_feats.json", "w"), indent=1)
