import base64, gzip, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
# Training session for the public 2.5D reader recipe (goodpjw2008) on our own account, not a submission:
# index the cache shards, check the cache against the public pipeline, then train one job per GPU. Built by build_train.py.
INPUT = os.environ.get("GPJW_INPUT", "/kaggle/input")
W = Path(os.environ.get("GPJW_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("GPJW_TMP", "/tmp"))
LOCAL = bool(os.environ.get("GPJW_LOCAL"))
JOBS = json.loads(base64.b64decode("__JOBS_B64__"))
N_PARTS = __N_PARTS__
RES, LOGS, SRC = W / "res", W / "logs", TMP / "own_src"
for d in (RES, LOGS, SRC):
    d.mkdir(parents=True, exist_ok=True)
for name, b64 in json.loads(base64.b64decode("__SRC_B64__")).items():
    (SRC / name).write_bytes(base64.b64decode(b64))
(TMP / "labels.csv").write_bytes(gzip.decompress(base64.b64decode("__LABELS_B64__")))
PORT_REF = "__PORT_B64__"


def walk(pred):
    out = []
    for d, dirs, files in os.walk(INPUT):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        out += [os.path.join(d, f) for f in files if pred(f)]
    return sorted(out)


# ---- 1. cache: the series are read straight from the packed shards on the input mount ----
# (unpacking them, 110 GB, is more than a session may write: the first attempt died without a log)
t0 = time.time()
shards = walk(lambda f: f.startswith("p") and "_s" in f and f.endswith(".npz"))
manifests = walk(lambda f: f.startswith("manifest_p") and f.endswith(".csv"))
assert len(manifests) == N_PARTS, f"expected {N_PARTS} cache parts, found {len(manifests)}: {manifests}"
meta = pd.concat(pd.read_csv(f, dtype={"study": str, "series": str}) for f in manifests)
assert not meta.error.fillna("").str.startswith("skipped").any(), "a cache part stopped at its output budget"
assert meta.series[meta.series.notna()].is_unique, "a series is in two cache parts"
meta.to_csv(TMP / "meta.csv", index=False)
index = {key: path for path in shards for key in np.load(path).files}
usable = meta[meta.n.fillna(0) >= 3]
missing = [s for s in usable.series if s not in index]
assert not missing, f"{len(missing)} usable series are in no shard, e.g. {missing[:3]}"
(TMP / "series_index.json").write_text(json.dumps(index))
train_csv = walk(lambda f: f == "train.csv")[0]
series_csv = str(Path(train_csv).parent / "train_series.csv")
lab = pd.read_csv(TMP / "labels.csv", index_col=0)
print(f"indexed {len(index)} series of {meta.study.nunique()} studies in {len(shards)} shards in {time.time() - t0:.0f}s; usable series {len(usable)}; "
      f"label studies without a usable series: {int((~lab.index.astype(str).isin(set(usable.study))).sum())}; "
      f"cache errors: {int(meta.error.notna().sum())}", flush=True)
env = {**os.environ, "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}

# ---- 2. the public checkpoint through this cache must reproduce the public DICOM path ----
backbone_auto = "convnext_tiny"
if PORT_REF and not LOCAL:
    (TMP / "port_ref.json").write_bytes(base64.b64decode(PORT_REF))
    ck = walk(lambda f: f == json.loads((TMP / "port_ref.json").read_text())["checkpoint"])[0]
    rc = subprocess.run([sys.executable, str(SRC / "port_check.py"), ck, str(TMP / "series_index.json"), str(TMP / "meta.csv"), series_csv,
                         str(TMP / "port_ref.json"), str(RES / "port_check.json")],
                        env={**env, "CUDA_VISIBLE_DEVICES": "0"}).returncode
    assert rc == 0, "the cache does not reproduce the public pipeline; no training was started"
    backbone_auto = json.loads((RES / "port_check.json").read_text())["checkpoint_args"]["backbone"]
    print("port check passed; the public checkpoint was trained with backbone", backbone_auto, flush=True)

# ---- 3. one training job per GPU ----
procs = []
for i, j in enumerate(JOBS):
    cmd = [sys.executable, "-u", str(SRC / j.get("script", "train_gpjw.py")), "--tag", j["tag"], "--fold", str(j["fold"]),
           "--backbone", backbone_auto if j["backbone"] == "auto" else j["backbone"], "--res", str(j["res"]),
           "--index", str(TMP / "series_index.json"), "--labels", str(TMP / "labels.csv"), "--meta", str(TMP / "meta.csv"),
           "--series-csv", series_csv, "--out", str(RES)] + [str(x) for x in j.get("extra", [])]
    if j.get("init"):                                # fine-tune from one of our earlier checkpoints
        cmd += ["--init", walk(lambda f, n=j["init"]: f == n)[0]]
    logf = open(LOGS / f"job_{j['tag']}_fold{j['fold']}.log", "w")
    print("job", j["tag"], "on GPU", i, ":", " ".join(cmd[3:]), flush=True)
    p = subprocess.Popen(cmd, env={**env, "CUDA_VISIBLE_DEVICES": str(i)}, stdout=logf, stderr=subprocess.STDOUT)
    if LOCAL:
        p.wait()                                   # one at a time on a machine without two GPUs
    procs.append((j, p, logf))
while any(p.poll() is None for _, p, _ in procs):
    time.sleep(5 if LOCAL else 300)
    for j, p, logf in procs:
        lines = (LOGS / f"job_{j['tag']}_fold{j['fold']}.log").read_text().strip().splitlines() or [""]
        key = ([l for l in lines if not l.startswith("  ")] or [""])[-1]        # last line that is not a step counter
        print(time.strftime("%H:%M"), j["tag"], "|", key[:200], "" if key == lines[-1] else "| " + lines[-1][:80], flush=True)
if PORT_REF and not LOCAL and __PUBLIC_OOF__:
    # the public checkpoints score their own folds: with our folds this completes the out-of-fold table
    rc = subprocess.run([sys.executable, str(SRC / "oof_public.py"), str(Path(ck).parent), str(TMP / "series_index.json"),
                         str(TMP / "meta.csv"), series_csv, str(TMP / "labels.csv"), str(RES)],
                        env={**env, "CUDA_VISIBLE_DEVICES": "0"}).returncode
    print("public out-of-fold predictions: rc", rc, flush=True)
bad = []
for j, p, logf in procs:
    logf.close()
    text = (LOGS / f"job_{j['tag']}_fold{j['fold']}.log").read_text()
    print(f"\n===== job {j['tag']} fold {j['fold']}: rc {p.returncode} =====")
    print("\n".join(l for l in text.splitlines() if not l.startswith("  "))[-6000:], flush=True)
    if p.returncode != 0:
        bad.append(j["tag"])
assert not bad, f"jobs failed: {bad}"
print("ALL JOBS DONE in", round((time.time() - t0) / 3600, 2), "h")
