import base64, io, json, multiprocessing, os, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd
# Training cache for the public 2.5D reader recipe (goodpjw2008): every train series as a canonical
# uint8 volume [n <= 64, 384, 384], made by the reader's own `load_series`, unchanged.
# One CPU job writes one part of the studies as a few compressed shard files (the output limit
# is 20 GB and a few hundred files). Built by build_cache.py.
PART, N_PARTS = __PART__, __N_PARTS__
SHARD = 48                 # studies per shard file
BUDGET = 18.5e9            # stop before the 20 GB output limit
INPUT = os.environ.get("GPJW_INPUT", "/kaggle/input")
W = Path(os.environ.get("GPJW_WORK", "/kaggle/working"))
TMP = Path(os.environ.get("GPJW_TMP", "/tmp"))
OUT = W / "cache384"
OUT.mkdir(parents=True, exist_ok=True)


def find(name, prefix=False):
    for d, dirs, files in os.walk(INPUT):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        for f in files:
            if f == name or (prefix and f.startswith(name)):
                return os.path.join(d, f)
    raise FileNotFoundError(name)


target, src = str(TMP / "_own_env"), TMP / "own_src"
if not os.environ.get("GPJW_LOCAL"):
    # the same OpenCV build the 0.943 pipeline installs, and the DICOM decoders the reader ships
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall",
                    find("opencv_python_headless-4.12.0.88", prefix=True)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
                    "--find-links", str(Path(find("cnxt_v0_fold0.pt")).parent),
                    "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
src.mkdir(parents=True, exist_ok=True)
for name, b64 in json.loads(base64.b64decode("__SRC_B64__")).items():
    (src / name).write_bytes(base64.b64decode(b64))
sys.path[:0] = [target, str(src)]
from preprocess import SIZE, load_series

train_csv = find("train.csv")
data_dir = str(Path(train_csv).parent)
root = f"{data_dir}/train_series"
studies = sorted(pd.read_csv(train_csv, dtype={"StudyInstanceUID": str}).StudyInstanceUID)
mine = studies[PART::N_PARTS]
se = pd.read_csv(f"{data_dir}/train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
by_study = {st: g for st, g in se[se.StudyInstanceUID.isin(set(mine))].groupby("StudyInstanceUID")}
shards = [(i // SHARD, mine[i:i + SHARD]) for i in range(0, len(mine), SHARD)]
print(f"part {PART}/{N_PARTS}: {len(mine)} of {len(studies)} studies, {len(shards)} shards", flush=True)


def written():
    return sum(f.stat().st_size for f in OUT.iterdir())


def do_shard(job):
    idx, sts = job
    if written() > BUDGET:
        return idx, [{"study": st, "series": "", "error": "skipped: output budget reached"} for st in sts]
    path, rows = OUT / f"p{PART}_s{idx:03d}.npz", []
    tmp = path.with_suffix(".part")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for st in sts:
            g = by_study.get(st)
            if g is None:
                rows.append({"study": st, "series": "", "error": "no rows in train_series.csv"})
                continue
            for sr, plane in zip(g.SeriesInstanceUID, g.Anatomical_Plane):
                info = {"study": st, "series": sr}
                try:
                    vol, extra = load_series(f"{root}/{st}/{sr}", plane)
                    buf = io.BytesIO()
                    np.save(buf, vol)
                    z.writestr(f"{sr}.npy", buf.getvalue())
                    info.update(extra)
                except Exception as e:
                    info["error"] = f"{type(e).__name__}: {e}"[:200]
                rows.append(info)
    os.replace(tmp, path)
    return idx, rows


t0, rows = time.time(), []
with multiprocessing.get_context("fork").Pool(4) as pool:      # fork: the workers reuse this cell's state
    for i, (idx, r) in enumerate(pool.imap_unordered(do_shard, shards)):
        rows.extend(r)
        print(f"shard {idx:3d} done ({i + 1}/{len(shards)})  {written() / 1e9:.2f} GB  {time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows)
for c in ("n", "n_raw", "n_bad", "plane", "slice_gap", "error"):
    if c not in df:
        df[c] = np.nan
df.to_csv(W / f"manifest_p{PART}.csv", index=False)
ok = df[df.n.fillna(0) >= 3]
raw = float(df.n.fillna(0).sum()) * SIZE * SIZE
summary = {"part": PART, "n_parts": N_PARTS, "studies": len(mine), "series": int((df.series != "").sum()),
           "series_usable": len(ok), "studies_with_usable_series": int(ok.study.nunique()),
           "errors": int(df.error.notna().sum()), "first_errors": df.error.dropna().unique()[:5].tolist(),
           "skipped_for_budget": int(df.error.fillna("").str.startswith("skipped").sum()),
           "slices_per_series": {"mean": float(ok.n.mean()), "median": float(ok.n.median()), "max": float(ok.n.max())},
           "bytes": written(), "raw_bytes": raw, "ratio": raw / max(written(), 1),
           "seconds": time.time() - t0, "seconds_per_study": (time.time() - t0) / max(len(mine), 1)}
(W / f"summary_p{PART}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
assert summary["skipped_for_budget"] == 0, "the output budget was reached: use more parts"
