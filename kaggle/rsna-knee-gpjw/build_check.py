"""Build rsna-knee-gpjw-check: the public 2.5D ConvNeXt reader (goodpjw2008, LB 0.929 alone)
on the same 300 training studies as our other check runs (not a submission).

The reader's three source files are taken unchanged from the public notebook in
research/public_notebooks/knee_g3/. One prediction file per fold checkpoint is written, so the
studies each checkpoint did not train on can be scored separately.

Usage: python build_check.py --user <kaggle-user>
"""
import argparse, ast, base64, json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
SLUG = "rsna-knee-gpjw-check"
NB = "rsna-knee-stack-2-5d-convnext-mil-lb-0-944"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--studies", default=str(RSNA / "runs/rsna-knee-solo-check/latest/check_studies.csv"))
a = ap.parse_args()
cells = json.loads((RSNA / f"research/public_notebooks/knee_g3/{NB}/{NB}.ipynb").read_text())["cells"]
src = {}
for c in cells:
    s = "".join(c["source"])
    if s.startswith("%%writefile /kaggle/working/own_src/"):
        head, body = s.split("\n", 1)
        ast.parse(body)
        src[head.rsplit("/", 1)[1].strip()] = base64.b64encode(body.encode()).decode()
assert sorted(src) == ["infer.py", "knee.py", "preprocess.py"], sorted(src)
studies = pd.read_csv(a.studies, dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
cell = (HERE / "gpjw_check_cell.py").read_text()
cell = cell.replace("__SRC_B64__", base64.b64encode(json.dumps(src).encode()).decode())
cell = cell.replace("__STUDIES_B64__", base64.b64encode(json.dumps(studies).encode()).decode())
ast.parse(cell)
out = HERE.parent / SLUG
out.mkdir(exist_ok=True)
nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
(out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
docker = json.loads((HERE.parent / "rsna-knee-base-repro" / "kernel-metadata.json").read_text())["docker_image"]
meta = {"id": f"{a.user}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader",
                            "mattiaangeli/opencv-python-headless-4120088-x86"],
        "kernel_sources": [], "model_sources": [], "docker_image": docker}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {SLUG}/: {len(studies)} studies, sources {sorted(src)}")
