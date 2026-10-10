"""Build rsna-knee-base-gold: the unchanged 0.943 pipeline run on the 58 expert-labelled
training studies (a check run, not a submission). One cell is put in front of the base
notebook; it points the pipeline's data root at a stand-in built from links.

Usage: python build.py --user <kaggle-user>
"""
import argparse, ast, json
from pathlib import Path

HERE = Path(__file__).parent
SLUG = "rsna-knee-base-gold"
ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
a = ap.parse_args()
base_dir = HERE.parent / "rsna-knee-base-repro"
nb = json.loads((base_dir / "rsna-knee-base-repro.ipynb").read_text())
cell = (HERE / "probe_cell.py").read_text()
ast.parse(cell)
n = len(nb["cells"])
nb["cells"] = [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell}] + nb["cells"]
out = HERE.parent / SLUG
out.mkdir(exist_ok=True)
(out / f"{SLUG}.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((base_dir / "kernel-metadata.json").read_text())
meta.update(id=f"{a.user}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {SLUG}/: 1 probe cell + {n} base cells (unchanged)")
