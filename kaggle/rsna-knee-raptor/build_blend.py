"""Build a submission notebook: the public single CoAtNet (nartaa, LB 0.949 alone, trained by its author with
OAI external data) unchanged, plus our own 2.5D reader rank-blended in at a fixed weight.

Cells: the reader's three source files as the public reader notebook writes them (unchanged); the single
model's two code cells (unchanged; they end by writing submission.csv); blend_cell.py (ours): runs the reader
with our checkpoints and writes rank((1 - w) * rank(single) + w * rank(reader)). A reader failure keeps the
single model's file.

Usage: python build_blend.py --user <kaggle-user> --slug rsna-knee-coat-calfg30 --weight 0.3 \
         --ours calFG_fold0.pt ... calFG_fold4.pt --from-kernels rsna-knee-gpjw-heads-a
"""
import argparse, ast, json
from pathlib import Path

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
SINGLE = "rsna-knee-0949-anatomical-mirror"
READER = "rsna-knee-stack-2-5d-convnext-mil-lb-0-944"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--slug", required=True)
ap.add_argument("--ours", nargs="+", required=True)
ap.add_argument("--from-kernels", nargs="*", default=[])
ap.add_argument("--weight", type=float, default=0.3, help="the reader's weight for every finding not named in --weight-finding")
ap.add_argument("--weight-finding", nargs="*", default=[], help='per-finding weights, e.g. "Lateral OA=1.0"')
a = ap.parse_args()
per_finding = {k: float(v) for k, v in (x.rsplit("=", 1) for x in a.weight_finding)}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
single = json.loads((RSNA / f"research/public_notebooks/knee_g4/{SINGLE}/{SINGLE}.ipynb").read_text())["cells"]
reader = json.loads((RSNA / f"research/public_notebooks/knee_g3/{READER}/{READER}.ipynb").read_text())["cells"]
writes = [c for c in reader if "".join(c["source"]).startswith("%%writefile /kaggle/working/own_src/")]
assert len(writes) == 3 and [c["cell_type"] for c in single] == ["markdown"] * 3 + ["code"] * 2
blend = (HERE / "blend_cell.py").read_text().replace("__WEIGHT_FINDING__", json.dumps(per_finding)).replace("__WEIGHT__", repr(a.weight)).replace("__OURS__", json.dumps(a.ours))
ast.parse(blend)
cells = ([code("import os\nos.makedirs('/kaggle/working/own_src', exist_ok=True)\n")] + [code("".join(c["source"])) for c in writes]
         + [code("".join(c["source"])) for c in single[3:]] + [code(blend)])
out = HERE.parent / a.slug
out.mkdir(exist_ok=True)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(out / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((RSNA / f"research/public_notebooks/knee_g4/{SINGLE}/kernel-metadata.json").read_text())
meta.pop("id_no", None)
meta.update(id=f"{a.user}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", is_private="true", enable_internet="false",
            dataset_sources=meta["dataset_sources"] + ["goodpjw2008/rsna-knee-2-5d-convnext-reader", "mattiaangeli/opencv-python-headless-4120088-x86"],
            kernel_sources=[f"{a.user}/{k}" for k in a.from_kernels])
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {a.slug}/: {len(cells)} cells; single model unchanged + reader {a.ours} at weight {a.weight}, per finding {per_finding}; datasets {meta['dataset_sources']}")
