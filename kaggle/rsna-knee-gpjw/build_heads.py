"""Build rsna-knee-gpjw-heads-<suffix>: a CPU session that retrains the head of every fold model on its saved
backbone features (feats_gpjw.py) under another label table and writes full checkpoints (fit_heads.py).
Not a submission.

  --job feats_file,checkpoint_file   once per fold, e.g. feats_ctrl_fold3.npz,ctrl_fold3.pt
  --labels                           label table (make_labels_calibrated.py), its name becomes the checkpoints' prefix
  --from-kernels                     our kernels that hold the feature files and our own checkpoints
  --extra                            further fit_heads.py arguments, e.g. "--seed 1"

Usage: python build_heads.py --user <kaggle-user> --suffix a --name calFG --labels ../../runs/solo/gpjw_labels_calFG.csv \
         --job feats_pub_fold0.npz,cnxt_v0_fold0.pt ... --from-kernels rsna-knee-gpjw-feats-a rsna-knee-gpjw-train-r1 ...
"""
import argparse, ast, base64, gzip, json, shlex
from pathlib import Path

from build_cache import notebook, reader_sources

HERE = Path(__file__).parent


def cell_for(jobs, labels_csv, name, extra=()):
    src = reader_sources(["knee.py"])
    text = (HERE / "fit_heads.py").read_text()
    ast.parse(text)
    src["fit_heads.py"] = base64.b64encode(text.encode()).decode()
    b64 = lambda obj: base64.b64encode(json.dumps(obj).encode()).decode()
    cell = (HERE / "heads_cell.py").read_text()
    cell = cell.replace("__JOBS_B64__", b64(jobs)).replace("__SRC_B64__", b64(src)).replace("__NAME__", name).replace("__EXTRA_B64__", b64(list(extra)))
    cell = cell.replace("__LABELS_B64__", base64.b64encode(gzip.compress(Path(labels_csv).read_bytes(), mtime=0)).decode())
    ast.parse(cell)
    return cell


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--job", action="append", required=True, help="feats_file,checkpoint_file")
    ap.add_argument("--from-kernels", nargs="+", required=True)
    ap.add_argument("--extra", default="")
    a = ap.parse_args()
    jobs = [dict(zip(("feats", "init"), spec.split(","))) for spec in a.job]
    assert len({j["feats"] for j in jobs}) == len(jobs)
    slug = f"rsna-knee-gpjw-heads-{a.suffix}"
    out = HERE.parent / slug
    out.mkdir(exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(notebook(cell_for(jobs, a.labels, a.name, shlex.split(a.extra))), indent=1))
    docker = json.loads((HERE.parent / "rsna-knee-base-repro" / "kernel-metadata.json").read_text())["docker_image"]
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "false", "enable_tpu": "false",
            "enable_internet": "false", "competition_sources": [],
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader"],
            "kernel_sources": [f"{a.user}/{k}" for k in a.from_kernels], "model_sources": [], "docker_image": docker}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {slug}/: {a.name} heads for", json.dumps(jobs))
