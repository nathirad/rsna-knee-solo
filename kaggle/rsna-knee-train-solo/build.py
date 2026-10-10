"""Build a training notebook for the solo member: one process per fold, one fold per GPU.

Kaggle's weekly quota counts session hours, not GPUs, so a T4 x2 session trains two folds
for the price of one. Five folds are three kernels: -a (folds 0,1), -b (2,3), -c (4).

Usage:
  python build.py --user <kaggle-user> --suffix a --folds 0,1
  python build.py --user <kaggle-user> --suffix a --folds 0,1 --labels pseudo --pseudo runs/solo/oof_round1.csv
  python build.py --user <kaggle-user> --suffix bench --folds 0 --bench        # 20 steps: time and memory
  python build.py --user <kaggle-user> --suffix ab --folds 0 --windows all,one # A/B: two variants, one per GPU

Each kernel writes to /kaggle/working/res: model_*.pt, member_*.json, oof_*.csv, gold_*.csv,
results_*.json. The solo submission notebook attaches these kernels as kernel_sources.
"""
import argparse, ast, base64, gzip, json
from pathlib import Path

HERE = Path(__file__).parent
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username that owns this kernel")
    ap.add_argument("--suffix", required=True, help="kernel slug suffix, e.g. a / b / c")
    ap.add_argument("--folds", required=True, help="comma list, e.g. 0,1")
    ap.add_argument("--cache-kernel", default=None, help="default <user>/rsna-knee-cache-9sl")
    ap.add_argument("--arch", default="cnx")
    ap.add_argument("--img", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--windows", default="all", help="all | one | all,one (one process per fold and mode)")
    ap.add_argument("--labels", default="pilk", choices=["pilk", "pseudo", "pseudo_keep"])
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--pseudo", default=None, help="out-of-fold table from research/oof_tools.py (labels=pseudo)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    if a.labels.startswith("pseudo") != bool(a.pseudo):
        ap.error("--labels pseudo / pseudo_keep need --pseudo <csv>, and --pseudo is only used with them")
    folds = [int(f) for f in a.folds.split(",")]
    modes = a.windows.split(",")
    if not set(modes) <= {"all", "one"}:
        ap.error("--windows takes all, one or all,one")
    jobs = [(f, m) for f in folds for m in modes]
    slug = f"rsna-knee-train-solo-{a.suffix}"
    cache_kernel = a.cache_kernel or f"{a.user}/rsna-knee-cache-9sl"

    job = (HERE / "train_solo.py").read_text()
    model = (HERE.parent / "rsna-knee-solo" / "solo_model.py").read_text()
    ast.parse(job); ast.parse(model)
    pseudo = b64(gzip.compress(Path(a.pseudo).read_bytes())) if a.pseudo else ""
    env = {"ARCH": a.arch, "IMG": str(a.img), "EPOCHS": str(a.epochs),
           "LABELS": a.labels, "ALPHA": str(a.alpha), "SEED": str(a.seed)}
    if a.bench:
        env["BENCH_ONLY"] = "1"
    cell = f'''import base64, gzip, os, subprocess, sys, time
from pathlib import Path
# Solo member training. Source: kaggle/rsna-knee-train-solo/ (build.py). Jobs (fold, windows): {jobs}.
JOBS, ENV = {jobs!r}, {env!r}
W = Path("/kaggle/working")


def find(name):
    """A file under /kaggle/input, without walking the DICOM trees."""
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if name in files:
            return os.path.join(d, name)
    raise FileNotFoundError(name)


IN = W / "in"; IN.mkdir(exist_ok=True)
for f in ["train.csv", "report_labels_v2.csv", "llm_labels_v4_blend.csv"]:
    if not (IN / f).exists():
        (IN / f).symlink_to(find(f))
cache = str(Path(find("cache_meta.json")).parent)
# The cache is read at random, a few MB at a time, by every fold process. Straight from the
# input mount that took ~10 s per optimizer step (bench v1). Copied once to local disk it
# is read at disk speed and the page cache is shared by the processes.
import shutil
from concurrent.futures import ThreadPoolExecutor
_t0, LOCAL = time.time(), Path("/tmp/solo_cache")
_files = [f for f in sorted(os.listdir(cache)) if f.endswith((".npy", ".json"))]
_need = sum(os.path.getsize(os.path.join(cache, f)) for f in _files)
_free = shutil.disk_usage("/tmp").free
print(f"cache {{_need / 1e9:.1f}} GB in {{len(_files)}} files; /tmp has {{_free / 1e9:.1f}} GB free", flush=True)
if _free > _need + 5e9:
    LOCAL.mkdir(exist_ok=True)
    with ThreadPoolExecutor(8) as _pool:
        list(_pool.map(lambda f: shutil.copyfile(os.path.join(cache, f), LOCAL / f), _files))
    cache = str(LOCAL)
    print(f"cache copied to {{cache}} in {{time.time() - _t0:.0f}} s", flush=True)
else:
    print("WARNING: not enough local disk for the cache; reading it from the input mount (slow)", flush=True)
(W / "train_solo.py").write_bytes(base64.b64decode("{b64(job.encode())}"))
(W / "solo_model.py").write_bytes(base64.b64decode("{b64(model.encode())}"))
PSEUDO = "{pseudo}"
if PSEUDO:
    (W / "pseudo_labels.csv").write_bytes(gzip.decompress(base64.b64decode(PSEUDO)))
    ENV["PSEUDO_CSV"] = str(W / "pseudo_labels.csv")

import torch
n_gpu = max(torch.cuda.device_count(), 1)
print("gpus", torch.cuda.device_count(), "cache", cache, flush=True)
running, failed, todo = [], [], list(JOBS)
while todo or running:
    while todo and len(running) < n_gpu:
        job = todo.pop(0)
        gpu = next(g for g in range(n_gpu) if g not in [r[2] for r in running])
        env = {{**os.environ, **ENV, "FOLD": str(job[0]), "WINDOWS": job[1], "IN_DIR": str(IN),
               "RES_DIR": str(W / "res"), "CACHE_DIR": cache, "CUDA_VISIBLE_DEVICES": str(gpu)}}
        running.append((job, subprocess.Popen([sys.executable, "-u", str(W / "train_solo.py")], env=env), gpu))
        print("started", job, "on gpu", gpu, flush=True)
    time.sleep(20)
    for r in list(running):
        if r[1].poll() is not None:
            running.remove(r)
            print("job", r[0], "rc", r[1].returncode, flush=True)
            if r[1].returncode != 0:
                failed.append(r[0])
assert not failed, f"jobs failed: {{failed}}"
print(sorted(os.listdir(W / "res")))
'''
    ast.parse(cell)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": cell}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE.parent / slug          # kaggle/<slug>/, like every other kernel folder
    out.mkdir(exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "true",
            "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "true",
            "machine_shape": "NvidiaTeslaT4",
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": ["pilkwang/rsna-knee-llm-labels", "stevenleehans/rsna-knee-llm-report-labels"],
            "kernel_sources": [cache_kernel], "model_sources": [], "docker_image": DOCKER}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {out.name}/: jobs (fold, windows) {jobs}, {env}, cache {cache_kernel}"
          + (f", pseudo table {a.pseudo}" if a.pseudo else ""))


if __name__ == "__main__":
    main()
