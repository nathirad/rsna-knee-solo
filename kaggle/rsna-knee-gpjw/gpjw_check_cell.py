import base64, glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Check run (not a submission): the public 2.5D ConvNeXt reader of goodpjw2008 (LB 0.929 alone)
# on 58 gold + 242 other training studies, one prediction file per fold checkpoint.
# Its source files are written unchanged from the public notebook. Built by build_check.py.
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


assets = str(Path(find("cnxt_v0_fold0.pt")).parent)
target = str(W / "_own_env")
# the same OpenCV build the 0.943 pipeline installs, and the DICOM decoders the reader ships
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall",
                find("opencv_python_headless-4.12.0.88", prefix=True)], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
                "--find-links", assets, "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
src = W / "own_src"
src.mkdir(exist_ok=True)
for name, b64 in json.loads(base64.b64decode("__SRC_B64__")).items():
    (src / name).write_bytes(base64.b64decode(b64))
sys.path[:0] = [target, str(src)]

import torch
from infer import DicomStudyDataset, load_models
from knee import LABELS, build_study_table
from preprocess import MAX_SLICES

assert LABELS == T
train_csv = find("train.csv")
data_dir = str(Path(train_csv).parent)
train = pd.read_csv(train_csv, dtype={"StudyInstanceUID": str})
gold = train.dropna(subset=T)
studies = json.loads(base64.b64decode("__STUDIES_B64__"))
assert set(gold.StudyInstanceUID) <= set(studies) and set(studies) <= set(train.StudyInstanceUID)
pd.DataFrame({"StudyInstanceUID": studies}).to_csv(W / "check_studies.csv", index=False)

t0 = time.time()
se = pd.read_csv(f"{data_dir}/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
se = se[se.StudyInstanceUID.isin(set(studies))]
root = f"{data_dir}/train_series"
n_files = {}
for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID):
    d = f"{root}/{st}/{sr}"
    n_files[sr] = min(len(os.listdir(d)), MAX_SLICES) if os.path.isdir(d) else 0
table = build_study_table(se, n_files)
ckpts = sorted(glob.glob(assets + "/cnxt_v0_fold*.pt"))
models = load_models(ckpts)                      # same loader, window count and batch size as infer.run
dl = torch.utils.data.DataLoader(DicomStudyDataset(studies, table, root, 12), batch_size=4, num_workers=4, pin_memory=True)
t1 = time.time()
out = [[] for _ in models]
with torch.no_grad():
    for i, (x, mask, pos) in enumerate(dl):
        x, mask, pos = x.cuda(non_blocking=True), mask.cuda(), pos.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            for j, (m, res) in enumerate(models):
                out[j].append(m(x, mask, pos, res=res).float().sigmoid().cpu())
        if i % 10 == 0:
            print(f"batch {i}/{len(dl)}  {time.time() - t1:.0f}s", flush=True)
t2 = time.time()
per_fold = {}
for path, o in zip(ckpts, out):
    name = Path(path).stem
    per_fold[name] = pd.DataFrame(torch.cat(o).numpy(), index=pd.Index(studies, name="StudyInstanceUID"), columns=T)
    per_fold[name].to_csv(W / f"gpjw_{name}.csv")
mean = sum(per_fold.values()) / len(per_fold)     # what infer.run returns: mean of fold probabilities
mean.to_csv(W / "gpjw_train_check.csv")

from sklearn.metrics import roc_auc_score
g = gold.set_index("StudyInstanceUID")[T].astype(int)
score = lambda p: {t: float(roc_auc_score(g[t], p.loc[g.index, t])) for t in T}
per = score(mean)
no_series = [s for s in studies if s not in table]
summary = {"checkpoints": [Path(p).name for p in ckpts], "n_studies": len(studies), "studies_without_series": len(no_series),
           "seconds": {"setup": t1 - t0, "predict": t2 - t1, "per_study": (t2 - t1) / len(studies)},
           "gold58_macro_auc": float(np.mean(list(per.values()))), "gold58_per_label": per,
           "gold58_macro_auc_per_fold": {k: float(np.mean(list(score(v).values()))) for k, v in per_fold.items()},
           "note": "the dataset card says the 58 gold studies were not trained on; the other 242 studies are in-fold for some checkpoints"}
(W / "gpjw_check_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
