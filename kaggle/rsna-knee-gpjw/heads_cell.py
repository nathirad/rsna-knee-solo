import base64, gzip, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Head session (CPU), not a submission: retrain the head of every fold model on its saved backbone features
# under another label table (fit_heads.py) and write full checkpoints in the public format. Built by build_heads.py.
INPUT = os.environ.get("GPJW_INPUT", "/kaggle/input")
W = Path(os.environ.get("GPJW_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("GPJW_TMP", "/tmp"))
JOBS = json.loads(base64.b64decode("__JOBS_B64__"))        # one per fold: the feature file and the checkpoint it came from
NAME, EXTRA = "__NAME__", json.loads(base64.b64decode("__EXTRA_B64__"))
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RES, SRC = W / "res", TMP / "own_src"
for d in (RES, SRC):
    d.mkdir(parents=True, exist_ok=True)
for name, b64 in json.loads(base64.b64decode("__SRC_B64__")).items():
    (SRC / name).write_bytes(base64.b64decode(b64))
(TMP / "labels.csv").write_bytes(gzip.decompress(base64.b64decode("__LABELS_B64__")))


def walk(name):
    out = []
    for d, dirs, files in os.walk(INPUT):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        out += [os.path.join(d, f) for f in files if f == name]
    return sorted(out)


t0 = time.time()
env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
for j in JOBS:
    feats, init = walk(j["feats"]), walk(j["init"])
    assert len(feats) == 1 and init, f"{j}: found features {feats}, checkpoint {init}"
    print(f"\n===== {j['feats']} + {j['init']} =====", flush=True)
    rc = subprocess.run([sys.executable, "-u", str(SRC / "fit_heads.py"), "--feats", feats[0], "--labels", str(TMP / "labels.csv"),
                         "--init", init[0], "--name", NAME, "--out", str(RES)] + [str(x) for x in EXTRA], env=env).returncode
    assert rc == 0, f"fit_heads.py failed for {j}"

# ---- the fold models together on the gold studies, with the old heads and with the new ones ----
from sklearn.metrics import roc_auc_score
lab = pd.read_csv(TMP / "labels.csv", index_col=0)
lab.index = lab.index.astype(str)
gold = lab.index[lab.is_gold == 1]
y = lab.loc[gold, [f"gold_{c}" for c in T]].values.astype(int)
old, new = [], []
for j in JOBS:
    z = np.load(walk(j["feats"])[0])
    st = [str(s) for s in z["studies"]]
    old.append(1 / (1 + np.exp(-z["logit"][[st.index(s) for s in gold]])))
    fold = j["feats"].rsplit("fold", 1)[1].split(".")[0]
    o = pd.read_csv(RES / f"oof_{NAME}_fold{fold}.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    new.append(o[o.set == "gold"].loc[gold, T].values)
each = lambda p: [roc_auc_score(y[:, i], p[:, i]) for i in range(len(T))]
a, b = each(np.mean(old, 0)), each(np.mean(new, 0))
print(f"\ngold-58, {len(JOBS)} fold models together: old heads {np.mean(a):.4f} -> new heads ({NAME}) {np.mean(b):.4f} ({np.mean(b) - np.mean(a):+.4f})")
print("per finding: " + " ".join(f"{c[:6]}={x:.3f}->{v:.3f}" for c, x, v in zip(T, a, b)))
json.dump({"name": NAME, "folds": len(JOBS), "gold_old_heads": float(np.mean(a)), "gold_new_heads": float(np.mean(b)),
           "each_old": dict(zip(T, map(float, a))), "each_new": dict(zip(T, map(float, b)))}, open(RES / f"summary_{NAME}.json", "w"), indent=1)
print("ALL HEADS DONE in", round((time.time() - t0) / 60, 1), "min")
