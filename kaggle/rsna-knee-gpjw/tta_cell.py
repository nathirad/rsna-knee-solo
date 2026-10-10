import base64, gzip, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Check run (not a submission): anatomical-mirror second view for the 2.5D reader, five fold models,
# each on its own held-out fold and the gold studies, through our cache. Built by build_tta_check.py.
W, TMP = Path("/kaggle/working"), Path("/tmp")
SRC = TMP / "own_src"
SRC.mkdir(parents=True, exist_ok=True)
for name, b64 in json.loads(base64.b64decode("__SRC_B64__")).items():
    (SRC / name).write_bytes(base64.b64decode(b64))
(TMP / "folds.csv").write_bytes(gzip.decompress(base64.b64decode("__FOLDS_B64__")))
CKPTS = __CKPTS__          # fold -> checkpoint file name


def walk(pred):
    out = []
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        out += [os.path.join(d, f) for f in files if pred(f)]
    return sorted(out)


t0 = time.time()
shards = walk(lambda f: f.startswith("p") and "_s" in f and f.endswith(".npz"))
meta = pd.concat(pd.read_csv(f, dtype={"study": str, "series": str}) for f in walk(lambda f: f.startswith("manifest_p") and f.endswith(".csv")))
meta.to_csv(TMP / "meta.csv", index=False)
index = {key: path for path in shards for key in np.load(path).files}
(TMP / "series_index.json").write_text(json.dumps(index))
series_csv = str(Path(walk(lambda f: f == "train.csv")[0]).parent / "train_series.csv")
args = [f"{k}={walk(lambda f, n=n: f == n)[0]}" for k, n in CKPTS.items()]
print(f"indexed {len(index)} series in {len(shards)} shards; checkpoints {args}", flush=True)
env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": "0"}
rc = subprocess.run([sys.executable, str(SRC / "tta_mirror.py"), str(TMP / "series_index.json"), str(TMP / "meta.csv"),
                     series_csv, str(TMP / "folds.csv"), str(W)] + args, env=env).returncode
print("rc", rc, "minutes", round((time.time() - t0) / 60, 1))
assert rc == 0
