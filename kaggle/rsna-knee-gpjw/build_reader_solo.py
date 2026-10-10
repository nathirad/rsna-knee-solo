"""Build a submission notebook that runs the 2.5D reader alone with our own checkpoints
(fast: no stack). Used to compare own training variants on the leaderboard's expert labels.

Usage: python build_reader_solo.py --user <kaggle-user> --slug rsna-knee-reader-ctrl34 --ours ctrl_fold3.pt ctrl_fold4.pt --from-kernels rsna-knee-gpjw-train-r1 rsna-knee-gpjw-train-r2
"""
import argparse, ast, json
from pathlib import Path

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
NB = "rsna-knee-stack-2-5d-convnext-mil-lb-0-944"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--ours", nargs="+", required=True)
ap.add_argument("--from-kernels", nargs="+", required=True)
a = ap.parse_args()
pub = json.loads((RSNA / f"research/public_notebooks/knee_g3/{NB}/{NB}.ipynb").read_text())["cells"]
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
cells = [code("import os\nos.makedirs('/kaggle/working/own_src', exist_ok=True)\n")]
writes = [c for c in pub if "".join(c["source"]).startswith("%%writefile /kaggle/working/own_src/")]
assert len(writes) == 3
cells += [code("".join(c["source"])) for c in writes]                 # preprocess.py, knee.py, infer.py: unchanged
run = (HERE / "reader_solo_cell.py").read_text().replace("__OURS__", json.dumps(a.ours))
ast.parse(run)
cells.append(code(run))
out = HERE.parent / a.slug
out.mkdir(exist_ok=True)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1))
docker = json.loads((HERE.parent / "rsna-knee-base-repro" / "kernel-metadata.json").read_text())["docker_image"]
meta = {"id": f"{a.user}/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "mattiaangeli/opencv-python-headless-4120088-x86"],
        "kernel_sources": [f"{a.user}/{k}" for k in a.from_kernels], "model_sources": [], "docker_image": docker}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {a.slug}/: reader alone with {a.ours}")
