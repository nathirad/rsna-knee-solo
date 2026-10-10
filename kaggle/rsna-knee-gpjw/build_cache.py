"""Build the CPU jobs that write the training cache of the public 2.5D reader recipe:
rsna-knee-gpjw-cache-p0 .. p{N-1}, one part of the train studies each (no GPU quota used).

The reader's `preprocess.py` is taken unchanged from the public notebook in
research/public_notebooks/knee_g3/.

Usage: python build_cache.py --user <kaggle-user> [--parts 6]
"""
import argparse, ast, base64, json
from pathlib import Path

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
NB = "rsna-knee-stack-2-5d-convnext-mil-lb-0-944"


def reader_sources(names):
    """The reader's source files as written by the public notebook, {name: base64}."""
    cells = json.loads((RSNA / f"research/public_notebooks/knee_g3/{NB}/{NB}.ipynb").read_text())["cells"]
    src = {}
    for c in cells:
        s = "".join(c["source"])
        if s.startswith("%%writefile /kaggle/working/own_src/"):
            head, body = s.split("\n", 1)
            name = head.rsplit("/", 1)[1].strip()
            if name in names:
                ast.parse(body)
                src[name] = base64.b64encode(body.encode()).decode()
    assert sorted(src) == sorted(names), sorted(src)
    return src


def notebook(cell):
    return {"cells": [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": cell}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 5}


def cell_for(part, parts):
    cell = (HERE / "cache_cell.py").read_text()
    cell = cell.replace("__SRC_B64__", base64.b64encode(json.dumps(reader_sources(["preprocess.py"])).encode()).decode())
    cell = cell.replace("__PART__", str(part)).replace("__N_PARTS__", str(parts))
    ast.parse(cell)
    return cell


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--parts", type=int, default=6)
    a = ap.parse_args()
    docker = json.loads((HERE.parent / "rsna-knee-base-repro" / "kernel-metadata.json").read_text())["docker_image"]
    for part in range(a.parts):
        slug = f"rsna-knee-gpjw-cache-p{part}"
        out = HERE.parent / slug
        out.mkdir(exist_ok=True)
        (out / f"{slug}.ipynb").write_text(json.dumps(notebook(cell_for(part, a.parts)), indent=1))
        meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
                "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
                "enable_internet": "false",
                "competition_sources": ["rsna-knee-abnormality-detection"],
                "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader",
                                    "mattiaangeli/opencv-python-headless-4120088-x86"],
                "kernel_sources": [], "model_sources": [], "docker_image": docker}
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built rsna-knee-gpjw-cache-p0..p{a.parts - 1}")
