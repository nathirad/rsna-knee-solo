import base64, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Check run (not a submission): the reader as the submission notebook runs it (its own infer.py as a
# subprocess, the timm build the stack ships, DICOM input) with FIVE checkpoints: the three public
# folds plus our folds 3 and 4, on the 58 gold studies. Built by build_reader5_check.py.
W = Path("/kaggle/working")
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OURS = __OURS__


def find(name, prefix=False):
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        for f in sorted(files):
            if f == name or (prefix and f.startswith(name)):
                return os.path.join(d, f)
    raise FileNotFoundError(name)


assets = str(Path(find("cnxt_v0_fold0.pt")).parent)
target, src = "/tmp/_own_env", Path("/tmp/own_src")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall",
                find("opencv_python_headless-4.12.0.88", prefix=True)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
                "--find-links", assets, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
timm_whl = find("timm-", prefix=True)                      # the timm build the stack ships, as in the notebook
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", target, timm_whl], check=False)
src.mkdir(parents=True, exist_ok=True)
for name, b64 in json.loads(base64.b64decode("__SRC_B64__")).items():
    (src / name).write_bytes(base64.b64decode(b64))

# a stand-in data root holding only the gold studies, so infer.py runs exactly as in the submission
train_csv = find("train.csv")
real = Path(train_csv).parent
train = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
gold = train.dropna(subset=T)
root = Path("/tmp/gold_root")
(root / "test_series").mkdir(parents=True, exist_ok=True)
for s in gold.StudyInstanceUID:
    if not (root / "test_series" / s).exists():
        os.symlink(real / "train_series" / s, root / "test_series" / s)
gold[["StudyInstanceUID"]].to_csv(root / "test.csv", index=False)
se = pd.read_csv(real / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
se[se.StudyInstanceUID.isin(set(gold.StudyInstanceUID))].to_csv(root / "test_series.csv", index=False)

ckpts = sorted(glob.glob(assets + "/cnxt_v0_fold*.pt")) + [find(n) for n in OURS]
print("checkpoints:", [f"{Path(c).name} ({os.path.getsize(c) / 1e6:.0f} MB)" for c in ckpts], "| timm wheel:", Path(timm_whl).name, flush=True)
env = {**os.environ, "PYTHONPATH": target + os.pathsep + os.environ.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": "0"}
t0 = time.time()
subprocess.run([sys.executable, str(src / "infer.py"), str(root), "test", str(W / "reader5_gold.csv")] + ckpts, check=True, env=env)
dt = time.time() - t0
pred = pd.read_csv(W / "reader5_gold.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
from sklearn.metrics import roc_auc_score
g = gold.set_index("StudyInstanceUID")[T].astype(int)
per = {t: float(roc_auc_score(g[t], pred.loc[g.index, t])) for t in T}
ver = subprocess.run([sys.executable, "-c", "import timm; print(timm.__version__)"], env=env, capture_output=True, text=True).stdout.strip()
summary = {"checkpoints": [Path(c).name for c in ckpts], "timm_in_reader_env": ver, "n_studies": len(g),
           "seconds": dt, "seconds_per_study": dt / len(g), "gold58_macro_auc": float(np.mean(list(per.values()))), "gold58_per_label": per}
(W / "reader5_check_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
