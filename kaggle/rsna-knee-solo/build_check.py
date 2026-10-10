"""Build rsna-knee-solo-check: our members on 300 training studies through the submission path.

Not a submission. It answers, on real data: are the decoded pixels identical to the training
cache, do the members reproduce their training-time gold predictions, what is gold-58 AUC
through this path, and how many seconds per study does it take.

Usage:
  python build_check.py --user <kaggle-user> --keep mode=one,labels=pilk \
      --kernel <user>/rsna-knee-train-solo-ab --kernel <user>/rsna-knee-train-solo-a --kernel <user>/rsna-knee-train-solo-b \
      --cache-kernel <owner>/rsna-knee-cache-9sl
"""
import argparse, ast, base64, json
from pathlib import Path

import build as solo_build

HERE = Path(__file__).parent
SLUG = "rsna-knee-solo-check"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--kernel", action="append", required=True, help="training kernels holding the members")
ap.add_argument("--cache-kernel", required=True, help="kernel holding the training cache (9sl)")
ap.add_argument("--keep", default="", help="member filter, e.g. mode=one,labels=pilk")
a = ap.parse_args()
script = "\n".join([solo_build.preproc("9sl"), "# %%\n" + (HERE / "solo_model.py").read_text(),
                    (HERE / "solo_driver.py").read_text()])
ast.parse(script)
cell = (HERE / "solo_check_cell.py").read_text()
cell = cell.replace("__SOLO_B64__", base64.b64encode(script.encode()).decode()).replace("__KEEP__", a.keep)
ast.parse(cell)
out = HERE.parent / SLUG
out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = {"id": f"{a.user}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"], "dataset_sources": [],
        "kernel_sources": a.kernel + [a.cache_kernel], "model_sources": [], "docker_image": solo_build.DOCKER}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {SLUG}/: members from {a.kernel}, filter {a.keep!r}, cache {a.cache_kernel}")
