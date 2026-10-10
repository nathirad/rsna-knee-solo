"""Build rsna-knee-gpjw-tta-check: anatomical-mirror second view for the 2.5D reader (check run, inference only).

Usage: python build_tta_check.py --user <kaggle-user>
"""
import argparse, ast, base64, gzip, json
from pathlib import Path

import pandas as pd

from build_cache import notebook, reader_sources

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
SLUG = "rsna-knee-gpjw-tta-check"
CKPTS = {"0": "cnxt_v0_fold0.pt", "1": "cnxt_v0_fold1.pt", "2": "cnxt_v0_fold2.pt", "3": "ctrl_fold3.pt", "4": "ctrl_fold4.pt"}

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--parts", type=int, default=6)
a = ap.parse_args()
src = reader_sources(["knee.py", "infer.py", "preprocess.py"])
for name in ("tta_mirror.py", "shards.py"):
    text = (HERE / name).read_text()
    ast.parse(text)
    src[name] = base64.b64encode(text.encode()).decode()
folds = pd.read_csv(RSNA / "runs/solo/gpjw_labels_v0.csv", dtype={"StudyInstanceUID": str})[["StudyInstanceUID", "is_gold", "fold"]]
cell = (HERE / "tta_cell.py").read_text()
cell = cell.replace("__SRC_B64__", base64.b64encode(json.dumps(src).encode()).decode())
cell = cell.replace("__FOLDS_B64__", base64.b64encode(gzip.compress(folds.to_csv(index=False).encode(), mtime=0)).decode())
cell = cell.replace("__CKPTS__", json.dumps(CKPTS))
ast.parse(cell)
out = HERE.parent / SLUG
out.mkdir(exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(notebook(cell), indent=1))
docker = json.loads((HERE.parent / "rsna-knee-base-repro" / "kernel-metadata.json").read_text())["docker_image"]
meta = {"id": f"{a.user}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader"],
        "kernel_sources": [f"{a.user}/rsna-knee-gpjw-cache-p{i}" for i in range(a.parts)]
                          + [f"{a.user}/rsna-knee-gpjw-train-r1", f"{a.user}/rsna-knee-gpjw-train-r2"],
        "model_sources": [], "docker_image": docker}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {SLUG}/")
