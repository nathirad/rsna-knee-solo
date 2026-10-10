"""Build rsna-knee-gpjw-train-<suffix>: one training session of the public 2.5D reader recipe
(goodpjw2008) on our own account. Not a submission. One job per GPU (T4 x2).

  --job tag,fold,backbone,res[,init]  repeat once per GPU; backbone "auto" = the public checkpoint's;
                                   init = one of our checkpoint files to fine-tune from (see --init-kernels)
  --budget-h                       hours each job may take (the session limit is 12 h)
  --extra                          further train_gpjw.py arguments for every job, e.g. "--micro 1"
  --script feats_gpjw.py           the jobs save backbone features of their init checkpoint instead of training
                                   (kernel rsna-knee-gpjw-feats-<suffix>; for head-only label experiments on a CPU)

Inputs of the session: the cache parts (rsna-knee-gpjw-cache-p*), the label table
runs/solo/gpjw_labels_v0.csv (make_labels_v0.py) and, for the cache check, the public checkpoints.

Usage: python build_train.py --user <kaggle-user> --suffix a --job ctrl,3,auto,256 --job cnv2,3,convnextv2_tiny.fcmae_ft_in22k_in1k,256
"""
import argparse, ast, base64, gzip, json, shlex
from pathlib import Path

import pandas as pd

from build_cache import notebook, reader_sources

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
     "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def port_reference(labels_csv, fold_csv=RSNA / "runs/rsna-knee-gpjw-check/latest/gpjw_cnxt_v0_fold0.csv"):
    """Predictions of the public fold-0 checkpoint on the gold studies through its own DICOM path."""
    lab = pd.read_csv(labels_csv, index_col=0)
    gold = lab.index[lab.is_gold == 1].astype(str).tolist()
    p = pd.read_csv(fold_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[gold, T]
    return {"checkpoint": "cnxt_v0_fold0.pt", "studies": gold, "pred": p.values.tolist()}


def cell_for(jobs, labels_csv, port_ref, parts=2, public_oof=True):
    src = reader_sources(["knee.py", "infer.py", "preprocess.py"])
    for name in ("train_gpjw.py", "feats_gpjw.py", "port_check.py", "oof_public.py", "shards.py"):
        text = (HERE / name).read_text()
        ast.parse(text)
        src[name] = base64.b64encode(text.encode()).decode()
    b64 = lambda obj: base64.b64encode(json.dumps(obj).encode()).decode()
    cell = (HERE / "train_cell.py").read_text()
    cell = cell.replace("__JOBS_B64__", b64(jobs)).replace("__SRC_B64__", b64(src)).replace("__N_PARTS__", str(parts))
    cell = cell.replace("__LABELS_B64__", base64.b64encode(gzip.compress(Path(labels_csv).read_bytes(), mtime=0)).decode())   # a notebook over ~1 MB is refused
    cell = cell.replace("__PORT_B64__", b64(port_ref) if port_ref else "")
    cell = cell.replace("__PUBLIC_OOF__", str(bool(public_oof)))
    ast.parse(cell)
    return cell


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--job", action="append", required=True, help="tag,fold,backbone,res")
    ap.add_argument("--parts", type=int, default=6)
    ap.add_argument("--budget-h", type=float, default=10.0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--init-kernels", nargs="*", default=[], help="our training kernels that hold the --job init checkpoints")
    ap.add_argument("--no-public-oof", action="store_true", help="skip scoring the public checkpoints on their own folds (already done once)")
    ap.add_argument("--labels", default=str(RSNA / "runs/solo/gpjw_labels_v0.csv"))
    ap.add_argument("--script", default="train_gpjw.py", choices=["train_gpjw.py", "feats_gpjw.py"])
    a = ap.parse_args()
    assert len(a.job) <= 2, "one job per GPU: at most two"
    jobs = []
    for spec in a.job:
        tag, fold, backbone, res, *init = spec.split(",")
        jobs.append({"tag": tag, "fold": int(fold), "backbone": backbone, "res": int(res),
                     "extra": ["--budget-h", str(a.budget_h)] + shlex.split(a.extra)})
        if init:
            jobs[-1]["init"] = init[0]
        if a.script != "train_gpjw.py":
            assert init, "a feature job needs the checkpoint to read: tag,fold,backbone,res,init"
            jobs[-1]["script"] = a.script
    assert len({(j["tag"], j["fold"]) for j in jobs}) == len(jobs), "two jobs would write the same files"
    slug = f"rsna-knee-gpjw-{'feats' if a.script == 'feats_gpjw.py' else 'train'}-{a.suffix}"
    out = HERE.parent / slug
    out.mkdir(exist_ok=True)
    (out / f"{slug}.ipynb").write_text(json.dumps(notebook(cell_for(jobs, a.labels, port_reference(a.labels), a.parts, not a.no_public_oof)), indent=1))
    docker = json.loads((HERE.parent / "rsna-knee-base-repro" / "kernel-metadata.json").read_text())["docker_image"]
    meta = {"id": f"{a.user}/{slug}", "title": slug, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
            "enable_internet": "true", "machine_shape": "NvidiaTeslaT4",
            "competition_sources": ["rsna-knee-abnormality-detection"],
            "dataset_sources": ["goodpjw2008/rsna-knee-2-5d-convnext-reader"],
            "kernel_sources": [f"{a.user}/rsna-knee-gpjw-cache-p{i}" for i in range(a.parts)] + [f"{a.user}/{k}" for k in a.init_kernels],
            "model_sources": [], "docker_image": docker}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"built {slug}/:", json.dumps(jobs))
