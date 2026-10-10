import base64, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Check run for our own members (not a submission). 58 gold + 242 other training studies go
# through the exact submission path (DICOM -> slots -> models), and the result is compared
# with what training saw: cached pixels, saved gold predictions, gold-58 AUC, seconds per study.
# Built by build_check.py.
W = Path("/kaggle/working")
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
KEEP = "__KEEP__"


def find_all(name):
    hits = []
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        hits += [os.path.join(d, f) for f in files if f == name]
    return hits


train = pd.read_csv([p for p in find_all("train.csv")][0], dtype={"StudyInstanceUID": str})
gold = train.dropna(subset=T)
other = train[~train.StudyInstanceUID.isin(gold.StudyInstanceUID)].sample(242, random_state=0)   # same 300 as the M448 check
pd.concat([gold[["StudyInstanceUID"]], other[["StudyInstanceUID"]]]).to_csv(W / "check_studies.csv", index=False)
(W / "_solo.py").write_bytes(base64.b64decode("__SOLO_B64__"))
t0 = time.time()
rc = subprocess.run([sys.executable, "-u", str(W / "_solo.py")],
                    env={**os.environ, "SOLO_SPLIT": "train", "SOLO_STUDY_CSV": str(W / "check_studies.csv"),
                         "SOLO_KEEP": KEEP, "SOLO_OUT": str(W / "solo_train_check.csv"),
                         "SOLO_DUMP": "/tmp/solo_dump.npz"}).returncode
print("rc", rc, "minutes", round((time.time() - t0) / 60, 1), flush=True)
assert rc == 0
rec = json.loads((W / "_solo_receipt.json").read_text())
d = np.load("/tmp/solo_dump.npz", allow_pickle=True)
studies, probs, members = [str(s) for s in d["studies"]], d["probs"], [str(m) for m in d["members"]]

# 1. pixels: what the submission path decodes vs the cache the members were trained on
cdir = Path([p for p in find_all("cache_meta.json")][0]).parent
meta = json.loads((cdir / "cache_meta.json").read_text())
parts = sorted(cdir.glob("cache_*_part*.npy"))
pos = {s: i for i, s in enumerate(meta["studies"])}
mask_tr = np.load(cdir / "mask.npy")
maps = [np.load(p, mmap_mode="r") for p in parts]
offs = np.cumsum([0] + [len(m) for m in maps])
same, worst, mask_same = 0, 0, 0
for k, s in enumerate(studies):
    i = pos[s]; p = int(np.searchsorted(offs, i, side="right") - 1)
    ref = np.asarray(maps[p][i - offs[p]])
    diff = int(np.abs(d["cache"][k].astype(np.int16) - ref.astype(np.int16)).max())
    same += diff == 0; worst = max(worst, diff)
    mask_same += bool(np.array_equal(d["mask"][k], mask_tr[i]))

# 2. predictions: each member here vs its own gold predictions saved by the training job
from sklearn.metrics import roc_auc_score
g = gold.set_index("StudyInstanceUID")[T].astype(int)
rows = [studies.index(s) for s in g.index]
pred_diff = {}
for k, name in enumerate(members):
    f = [p for p in find_all(name.replace("model_", "gold_").replace(".pt", ".csv"))][0]
    tg = pd.read_csv(f, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[g.index, T].values
    pred_diff[name] = float(np.abs(probs[k][rows] - tg).max())

# 3. gold-58 through the submission path
ens = sum(pd.DataFrame(p[rows], columns=T).rank(pct=True) for p in probs) / len(probs)
per = {t: float(roc_auc_score(g[t].values, ens[t].values)) for t in T}
n = len(studies)
out = {"members": members, "n_studies_asked": int(rec["n_studies"]), "n_without_prediction": int(rec["n_without_prediction"]),
       "pixels": {"studies_identical_to_training_cache": int(same), "of": n, "max_abs_diff": int(worst),
                  "slot_masks_identical": int(mask_same)},
       "prediction_max_abs_diff_vs_training": pred_diff,
       "gold58_macro_auc_submission_path": float(np.mean(list(per.values()))), "gold58_per_label": per,
       "seconds": rec["seconds"],
       "seconds_per_study": {k: v / n for k, v in rec["seconds"].items()},
       "projected_minutes_1300_studies": rec["seconds"]["total_script"] / n * 1300 / 60}
(W / "solo_check_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
