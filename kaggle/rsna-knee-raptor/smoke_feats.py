"""Local plumbing test of the feature script for the public single CoAtNet (no GPU, no real weights).

The script is the public notebook's cell with extract_main.py appended (build_feats.script()). Here it is
imported, pointed at a synthetic DICOM corpus, and given a small stand-in model in place of the real
checkpoint (which lives on Kaggle); then extract_features() runs as it will there. Checked: one row per
study in the requested order, feature shape [N, 94, F], and head(saved features) == saved logits.

Run: uv run --no-project --python 3.12 --with torch --with timm --with pydicom --with pandas --with scikit-learn \
       --with opencv-python-headless python kaggle/rsna-knee-raptor/smoke_feats.py
"""
import os, sys, tempfile, types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "rsna-knee-solo")]
import build_feats                       # noqa: E402
import smoke_local as corpus             # noqa: E402  (synthetic DICOM corpus)

tmp = Path(tempfile.mkdtemp(prefix="raptor_smoke_"))
comp = tmp / "comp"
comp.mkdir()
studies, _, _, _ = corpus.make_corpus(comp)
ids = studies[:6]
pd.DataFrame({"StudyInstanceUID": ids}).to_csv(tmp / "studies.csv", index=False)
os.environ.update(RAPTOR_SPLIT="train", RAPTOR_STUDIES=str(tmp / "studies.csv"), RAPTOR_OUT=str(tmp / "res"))
mod = types.ModuleType("raptor_feats")
mod.__dict__["__file__"] = str(tmp / "raptor_feats.py")     # the script writes its DICOM reader module next to itself
exec(compile(build_feats.script(), "raptor_feats.py", "exec"), mod.__dict__)

import timm, torch                        # noqa: E402
torch.manual_seed(0)
mod.find_test_root = lambda: str(comp)
mod.find_weight_file = lambda name: name


def stand_in(path, arch, res, device, ngpu=1):
    bb = timm.create_model("convnext_atto", pretrained=False, num_classes=0)
    return mod.RaptorClassifier(bb, F_dim=bb.num_features).eval().to(device), mod.INPUT_RES


mod.load_model = stand_in


class SerialPool:
    """Stand-in for the fork pool: on macOS a fork after torch has started its threads hangs. Same calls, one process."""

    def __init__(self, n, initializer=None):
        initializer and initializer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def imap_unordered(self, fn, items, chunksize=1):
        return map(fn, items)


import multiprocessing                    # noqa: E402
multiprocessing.get_context = lambda kind: types.SimpleNamespace(Pool=SerialPool)
mod.extract_features()
z = np.load(tmp / "res" / "feats_raptor_train.npz")
assert [str(s) for s in z["studies"]] == ids, "studies are not in the requested order"
assert z["feat"].shape[:2] == (len(ids), mod.K_EVAL) and z["feat"].dtype == np.float16, z["feat"].shape
torch.manual_seed(0)
model, _ = stand_in(None, None, None, "cpu")
with torch.no_grad():
    again = model.head(torch.from_numpy(z["feat"].astype(np.float32))).numpy()
diff = float(np.abs(again - z["logit"]).max())
assert diff < 5e-3, f"head(saved features) differs from the saved logits: {diff}"
print(f"features {z['feat'].shape}, slices filled per study {z['filled'].tolist()}, head(features) == saved logits ({diff:.1e})")
print("RAPTOR FEATURE SMOKE PASSED")
