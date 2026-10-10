"""Build rsna-knee-gpjw-reader5-check: the reader exactly as the submission notebook runs it, with the
three public fold checkpoints plus our own, on the 58 gold studies (not a submission).

Usage: python build_reader5_check.py --user <kaggle-user> --ours ctrl_fold3.pt ctrl_fold4.pt --from-kernels rsna-knee-gpjw-train-r1 rsna-knee-gpjw-train-r2
"""
import argparse, ast, base64, json
from pathlib import Path

from build_cache import notebook, reader_sources

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
NB = "rsna-knee-stack-2-5d-convnext-mil-lb-0-944"
SLUG = "rsna-knee-gpjw-reader5-check"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--ours", nargs="+", required=True, help="our checkpoint file names")
ap.add_argument("--from-kernels", nargs="+", required=True, help="our training kernels that hold them")
a = ap.parse_args()
cell = (HERE / "reader5_check_cell.py").read_text()
cell = cell.replace("__SRC_B64__", base64.b64encode(json.dumps(reader_sources(["knee.py", "infer.py", "preprocess.py"])).encode()).decode())
cell = cell.replace("__OURS__", json.dumps(a.ours))
ast.parse(cell)
pub = json.loads((RSNA / f"research/public_notebooks/knee_g3/{NB}/kernel-metadata.json").read_text())
out = HERE.parent / SLUG
out.mkdir(exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(notebook(cell), indent=1))
meta = {"id": f"{a.user}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
        "enable_internet": "false", "machine_shape": "NvidiaTeslaT4",
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": pub["dataset_sources"],          # the public notebook's datasets: timm and OpenCV wheels, reader weights
        "kernel_sources": [f"{a.user}/{k}" for k in a.from_kernels], "model_sources": [], "docker_image": pub["docker_image"]}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {SLUG}/: public folds + {a.ours}")
