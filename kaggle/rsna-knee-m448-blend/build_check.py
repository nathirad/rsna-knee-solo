"""Build rsna-knee-m448-check: the M448 reader alone on 300 training studies (not a submission).

It answers three things the three visible test studies cannot: seconds per study, how many
studies the reader cannot read (the blend needs 98% coverage), and its gold-58 AUC.

Usage: python build_check.py --user <kaggle-user>
"""
import argparse, ast, base64, json
from pathlib import Path

HERE = Path(__file__).parent
SLUG = "rsna-knee-m448-check"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
a = ap.parse_args()
leg = (HERE / "m448_leg.py").read_text()
ast.parse(leg)
cell = (HERE / "m448_check_cell.py").read_text().replace("__LEG_B64__", base64.b64encode(leg.encode()).decode())
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
        "dataset_sources": ["mattiaangeli/rsna-knee-cnx-m448-f0-public",
                            "mattiaangeli/opencv-python-headless-4120088-x86"],
        "kernel_sources": [], "model_sources": [], "docker_image": docker}
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {SLUG}/")
