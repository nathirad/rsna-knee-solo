"""Extra reader for the 0.943 public pipeline: the public ConvNeXt-S M448 fold-0 model.

Dataset: mattiaangeli/rsna-knee-cnx-m448-f0-public (weights + inference code by the author of
the 0.943 recipe; the 0.943 pipeline itself does not use it). This script runs in its own
process after the public pipeline has written submission.csv, and writes one CSV of
probabilities. It calls the dataset's own functions; the only change from the dataset's
`run()` is that a study that cannot be read gets an empty row instead of ending the leg.
"""
import json, os, sys, time
from pathlib import Path

T0 = time.time()
ROOT = Path(os.environ["M448_ROOT"])
OUT = Path(os.environ["M448_OUT"])
BUDGET_S = float(os.environ.get("M448_BUDGET_S", "5400"))
# Scoring always reads the test split. `train` + a study list is for check runs only
# (timing and readability on more than the three visible test studies).
SPLIT = os.environ.get("M448_SPLIT", "test")
STUDY_CSV = os.environ.get("M448_STUDY_CSV", "")
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import torch
import btkr_cnx_montage448_optional as m


def log(msg):
    print(f"[m448 {time.time() - T0:7.1f}s] {msg}", flush=True)


manifest = json.loads((ROOT / "btkr_cnx_montage448_manifest.json").read_text())
workers = max(1, min(4, os.cpu_count() or 1))
cv2.setNumThreads(1)
torch.set_num_threads(workers)
data = m.find_competition_root()
labels = list(m.LABELS)
if SPLIT == "test":
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"StudyInstanceUID": str})
    assert sample.columns.tolist() == ["StudyInstanceUID", *labels], "submission schema drift"
else:
    sample = pd.read_csv(STUDY_CSV, dtype={"StudyInstanceUID": str})[["StudyInstanceUID"]]
    sample[labels] = np.nan
studies = sample.StudyInstanceUID.astype(str).tolist()
log(f"{len(studies)} {SPLIT} studies; scanning series headers")
selected = m.build_raw_manifest(data, SPLIT, workers=workers, studies=set(studies))
t_scan = time.time() - T0
model = m.load_model(ROOT / manifest["checkpoint"]["file"], manifest["checkpoint"]["sha256"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    model = model.to(device, memory_format=torch.channels_last).eval()
else:
    model = model.to(device).eval()
t_ready = time.time() - T0
log(f"header scan {t_scan:.0f}s, model ready on {device} at {t_ready:.0f}s; budget {BUDGET_S:.0f}s")

preds, errors, timed_out = [], [], False
t_loop = time.time()
for k, study in enumerate(studies):
    if time.time() - T0 > BUDGET_S:
        timed_out = True
        log(f"time budget reached at study {k}/{len(studies)}")
        break
    try:
        batch = m.build_raw_study_batch(data, SPLIT, study, selected, workers=workers)
        preds.append(m.predict_study(model, batch, device, micro=4))
    except Exception as exc:                       # noqa: BLE001 - counted and reported
        errors.append({"study": study, "error": f"{type(exc).__name__}: {exc}"[:200]})
        preds.append(np.full(len(labels), np.nan, np.float32))
    if (k + 1) % 100 == 0:
        log(f"{k + 1}/{len(studies)} studies, {(time.time() - t_loop) / (k + 1):.2f} s/study, {len(errors)} unreadable")
n_done = len(preds)
preds += [np.full(len(labels), np.nan, np.float32)] * (len(studies) - n_done)
result = sample.copy()
result[labels] = np.stack(preds)
result.to_csv(OUT, index=False)
ok = int(np.isfinite(result[labels].to_numpy()).all(axis=1).sum())
receipt = {"n_studies": len(studies), "n_predicted": ok, "n_unreadable": len(errors),
           "timed_out": timed_out, "first_errors": errors[:5],
           "seconds": {"header_scan": t_scan, "ready": t_ready, "studies": time.time() - t_loop,
                       "per_study": (time.time() - t_loop) / max(n_done, 1), "total": time.time() - T0},
           "checkpoint": manifest["checkpoint"]["file"], "device": str(device)}
OUT.with_suffix(".json").write_text(json.dumps(receipt, indent=1))
log(f"wrote {OUT}: {ok}/{len(studies)} predicted, {receipt['seconds']['per_study']:.2f} s/study")
