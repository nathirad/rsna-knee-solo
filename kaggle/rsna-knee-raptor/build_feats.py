"""Build rsna-knee-raptor-feats: backbone features of the public single CoAtNet (nartaa, LB 0.949 alone)
for every train study, so that its head can be retrained on a CPU under another label table
(the head-only method of ../rsna-knee-gpjw/fit_heads.py and research/head_lab.py). Not a submission.

The script the session runs is the public notebook's code cell, unchanged except that its final
`main()` call is dropped, followed by extract_main.py. The checkpoint was trained by its author with
OAI external data; see runs/LOG.md (2026-10-09) for the host's statements on that.

Usage: python build_feats.py --user <kaggle-user> [--limit 300] [--budget-h 2.5]
"""
import argparse, ast, base64, gzip, json
from pathlib import Path

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
NB = "rsna-knee-0949-anatomical-mirror"
TAIL = '\n\nif __name__ == "__main__":\n    main()\n'


def script(nb=NB):
    src = RSNA / f"research/public_notebooks/knee_g4/{nb}/{nb}.ipynb"
    cell = "".join(json.loads(src.read_text())["cells"][4]["source"])
    assert cell.rstrip().endswith(TAIL.strip()), "the public cell no longer ends with its main() call"
    body = cell.rstrip()[:-len(TAIL.strip())].rstrip() + "\n"
    text = body + (HERE / "extract_main.py").read_text()
    ast.parse(text)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--slug", default="rsna-knee-raptor-feats")
    ap.add_argument("--limit", type=int, default=0, help="only the first N studies (gold first); 0 = all")
    ap.add_argument("--budget-h", type=float, default=2.5)
    ap.add_argument("--labels", default=str(RSNA / "runs/solo/gpjw_labels_v0.csv"))
    ap.add_argument("--notebook", default=NB, help="public notebook whose cell is used, e.g. rsna-knee-0945-efficient-224crop (the author's second model)")
    a = ap.parse_args()
    cell = (HERE / "feats_cell.py").read_text()
    cell = cell.replace("__SCRIPT_B64__", base64.b64encode(script(a.notebook).encode()).decode())
    cell = cell.replace("__LABELS_B64__", base64.b64encode(gzip.compress(Path(a.labels).read_bytes(), mtime=0)).decode())
    cell = cell.replace("__LIMIT__", str(a.limit)).replace("__BUDGET_H__", str(a.budget_h))
    ast.parse(cell)
    out = HERE.parent / a.slug
    out.mkdir(exist_ok=True)
    nb = {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    (out / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((RSNA / f"research/public_notebooks/knee_g4/{a.notebook}/kernel-metadata.json").read_text())
    meta.pop("id_no", None)
    meta.update(id=f"{a.user}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", is_private="true", enable_internet="false")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {a.slug}/: {len(cell)} chars, limit {a.limit or 'all'}, budget {a.budget_h} h; datasets {meta.get('dataset_sources')}, "
          f"machine {meta.get('machine_shape')}, docker {str(meta.get('docker_image'))[:60]}")
