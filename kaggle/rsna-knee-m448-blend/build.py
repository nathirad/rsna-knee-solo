"""Build rsna-knee-m448-blend.ipynb = the 0.943 base notebook (unchanged) + one extra public reader.

The extra reader is mattiaangeli/rsna-knee-cnx-m448-f0-public (ConvNeXt-S, 448 px montages,
fold 0), rank-mixed into the base result with one global weight and a fallback to the base
file. Same pattern as ../rsna-knee-blend/, with a public model instead of our own leg.

Usage: python build.py --user <kaggle-user> [--weight 0.10]
"""
import argparse, ast, base64, json
from pathlib import Path

HERE = Path(__file__).parent
SLUG = "rsna-knee-m448-blend"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="Kaggle username that owns this kernel")
    ap.add_argument("--weight", default="0.10", help="rank weight of the extra reader")
    a = ap.parse_args()
    float(a.weight)
    base_dir = HERE.parent / "rsna-knee-base-repro"
    base = json.loads((base_dir / "rsna-knee-base-repro.ipynb").read_text())
    leg = (HERE / "m448_leg.py").read_text()
    ast.parse(leg)
    src = (HERE / "m448_cells.py").read_text()
    src = src.replace("__LEG_B64__", base64.b64encode(leg.encode()).decode()).replace("__M448_W__", a.weight)
    cells = []
    for chunk in src.split("# %%")[1:]:
        if chunk.startswith(" [markdown]"):
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in chunk.split("\n")[1:]).strip()
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text})
        else:
            code = chunk.split("\n", 1)[1].strip() + "\n"
            ast.parse(code)
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": code})
    n_base = len(base["cells"])
    base["cells"] = base["cells"] + cells
    (HERE / f"{SLUG}.ipynb").write_text(json.dumps(base, indent=1))
    meta = json.loads((base_dir / "kernel-metadata.json").read_text())
    meta.update(id=f"{a.user}/{SLUG}", title=SLUG, code_file=f"{SLUG}.ipynb")
    meta["dataset_sources"] = meta["dataset_sources"] + ["mattiaangeli/rsna-knee-cnx-m448-f0-public"]
    (HERE / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {SLUG}.ipynb: {n_base} base cells (unchanged) + {len(cells)}, weight {a.weight}, "
          f"{len(meta['dataset_sources'])} datasets")


if __name__ == "__main__":
    main()
