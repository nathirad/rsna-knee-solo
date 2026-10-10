"""Build rsna-knee-solo.ipynb: our own members alone -> submission.csv (no public ensemble).

The script the notebook runs is: preprocessing (../rsna-knee-blend/leg_preproc.py, patched
to the cache contract of the members) + solo_model.py + solo_driver.py.

Usage:
  python build.py --user <kaggle-user> --contract e3  --dataset <user>/rsna-knee-own-members --archs cnx
  python build.py --user <kaggle-user> --contract 9sl --kernel <user>/rsna-knee-train-solo-a \
                  --kernel <user>/rsna-knee-train-solo-b --kernel <user>/rsna-knee-train-solo-c

--contract names the cache the members were trained on:
  e3   rsna-knee-cache-full: 336 px, 3 slices, band 0.20-0.80 (the E3 ConvNeXt-T / DINO weights)
  9sl  rsna-knee-cache-9sl:  256 px, 9 slices, band 0.15-0.85 (members from rsna-knee-train-solo)
The driver re-checks it against every member at run time and stops on a mismatch.
"""
import argparse, ast, base64, json
from pathlib import Path

HERE = Path(__file__).parent
CONTRACTS = {"e3": {"img": 336, "groups": 1, "band": (0.20, 0.80)},
             "9sl": {"img": 256, "groups": 3, "band": (0.15, 0.85)}}
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"


def patch_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"expected exactly one {old!r} in leg_preproc.py, found {text.count(old)}")
    return text.replace(old, new)


def preproc(contract):
    c = CONTRACTS[contract]
    t = (HERE.parent / "rsna-knee-blend" / "leg_preproc.py").read_text()
    t = patch_once(t, "CACHE_IMG = 336\n", f"CACHE_IMG = {c['img']}\n")
    t = patch_once(t, "N_GROUP_MAX = 1\n", f"N_GROUP_MAX = {c['groups']}\n")
    t = patch_once(t, "SLICE_BAND = (0.20, 0.80)", f"SLICE_BAND = ({c['band'][0]:.2f}, {c['band'][1]:.2f})")
    # plan_cache() lowers the slice count when memory is short, sized for a training corpus
    # that this notebook never caches. A member needs the slice count it was fitted on, so
    # here it is fixed (1,300 test studies at 256 px x 9 slices are 4.6 GB).
    t = patch_once(t, 'N_GROUP = plan_cache(len(pd.read_csv(ROOT / "train.csv")),\n'
                      '                     len(pd.read_csv(ROOT / "test.csv")))\n',
                   "N_GROUP = N_GROUP_MAX     # solo: fixed by the cache contract, not by free memory\n")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username that owns this kernel")
    ap.add_argument("--contract", choices=sorted(CONTRACTS), required=True)
    ap.add_argument("--dataset", action="append", default=[], help="dataset holding members (repeatable)")
    ap.add_argument("--kernel", action="append", default=[], help="training kernel whose output holds members (repeatable)")
    ap.add_argument("--archs", default="", help="only run members of these archs, e.g. cnx")
    ap.add_argument("--slug", default="rsna-knee-solo")
    a = ap.parse_args()
    if not (a.dataset or a.kernel):
        ap.error("give at least one --dataset or --kernel that holds the members")

    script = "\n".join([preproc(a.contract),
                        "# %%\n" + (HERE / "solo_model.py").read_text(),
                        (HERE / "solo_driver.py").read_text()])
    ast.parse(script)
    (HERE / "_solo.py").write_text(script)
    env = {"SOLO_ARCHS": a.archs} if a.archs else {}
    cell = f'''import base64, json, os, subprocess, sys, time
from pathlib import Path
# Our own members alone, no public ensemble. Script source: kaggle/rsna-knee-solo/ (build.py).
_t0 = time.time()
Path("/kaggle/working/_solo.py").write_bytes(base64.b64decode("{base64.b64encode(script.encode()).decode()}"))
rc = subprocess.run([sys.executable, "-u", "/kaggle/working/_solo.py"],
                    env={{**os.environ, **{env!r}}}).returncode
print("rc", rc, "wall seconds", round(time.time() - _t0), flush=True)
assert rc == 0, "solo script failed; no submission.csv was written"
print(open("/kaggle/working/_solo_receipt.json").read())
print(open("/kaggle/working/submission.csv").read()[:1200])
'''
    ast.parse(cell)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                     "source": cell}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (HERE / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.ipynb",
            "language": "python", "kernel_type": "notebook", "is_private": "true",
            "enable_gpu": "true", "enable_tpu": "false", "enable_internet": "false",
            "machine_shape": "NvidiaTeslaT4",
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": a.dataset, "kernel_sources": a.kernel, "model_sources": [],
            "docker_image": DOCKER}
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {a.slug}.ipynb: contract {a.contract} {CONTRACTS[a.contract]}, script {len(script)} chars, "
          f"members from {a.dataset + a.kernel}")


if __name__ == "__main__":
    main()
