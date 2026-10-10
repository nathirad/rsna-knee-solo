import base64, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Check run for the M448 reader (not a submission): seconds per study, unreadable studies and
# gold-58 AUC on 58 gold + 242 other training studies. Built by build_check.py.
W = Path("/kaggle/working")
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def find(name, prefix=False):
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        for f in files:
            if f == name or (prefix and f.startswith(name)):
                return os.path.join(d, f)
    raise FileNotFoundError(name)


# the same OpenCV build the 0.943 pipeline installs, so the rendering matches the scored run
whl = find("opencv_python_headless-4.12.0.88", prefix=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                "--force-reinstall", whl], check=True)
train = pd.read_csv(find("train.csv"), dtype={"StudyInstanceUID": str})
gold = train.dropna(subset=T)
other = train[~train.StudyInstanceUID.isin(gold.StudyInstanceUID)].sample(242, random_state=0)
pd.concat([gold[["StudyInstanceUID"]], other[["StudyInstanceUID"]]]).to_csv(W / "check_studies.csv", index=False)
(W / "_m448_leg.py").write_bytes(base64.b64decode("__LEG_B64__"))
root = str(Path(find("btkr_cnx_montage448_manifest.json")).parent)
t0 = time.time()
rc = subprocess.run([sys.executable, "-u", str(W / "_m448_leg.py")],
                    env={**os.environ, "M448_ROOT": root, "M448_OUT": str(W / "m448_train_check.csv"),
                         "M448_SPLIT": "train", "M448_STUDY_CSV": str(W / "check_studies.csv"),
                         "M448_BUDGET_S": "7200"}).returncode
print("rc", rc, "minutes", round((time.time() - t0) / 60, 1), flush=True)
assert rc == 0
rec = json.loads((W / "m448_train_check.json").read_text())
pred = pd.read_csv(W / "m448_train_check.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
from sklearn.metrics import roc_auc_score
g = gold.set_index("StudyInstanceUID")[T].astype(int)
pg = pred.loc[g.index, T]
ok = pg.notna().all(axis=1)
per = {t: float(roc_auc_score(g.loc[ok, t], pg.loc[ok, t])) for t in T}
out = {"receipt": rec, "gold58_n": int(ok.sum()), "gold58_macro_auc": float(np.mean(list(per.values()))),
       "gold58_per_label": per,
       "note": "gold-58 may be in this checkpoint's training data, so this AUC is an upper bound, not a validation score"}
(W / "m448_check_summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
