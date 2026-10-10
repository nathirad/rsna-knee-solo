# Check run, not a submission: the unchanged 0.943 pipeline is pointed at training studies
# that carry expert labels, so its predictions can be scored and compared with our members.
# The pipeline resolves its data root from RSNA_COMP_ROOT; this cell builds a stand-in root
# in /tmp (links only, nothing is copied) and sets that variable before the pipeline starts.
import os
from pathlib import Path
import pandas as pd

_P_T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
        "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_P_REAL = next(p for p in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
                           Path("/kaggle/input/rsna-knee-abnormality-detection")] if (p / "train.csv").is_file())
_p_train = pd.read_csv(_P_REAL / "train.csv", dtype={"StudyInstanceUID": str})
_p_ids = _p_train.dropna(subset=_P_T).StudyInstanceUID.tolist()          # the 58 expert-labelled studies
_P_FAKE = Path("/tmp/probe_root")
(_P_FAKE / "test_series").mkdir(parents=True, exist_ok=True)
for _s in _p_ids:
    if not (_P_FAKE / "test_series" / _s).exists():
        os.symlink(_P_REAL / "train_series" / _s, _P_FAKE / "test_series" / _s)
pd.DataFrame({"StudyInstanceUID": _p_ids}).to_csv(_P_FAKE / "test.csv", index=False)
_p_ser = pd.read_csv(_P_REAL / "train_series.csv", dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
_p_ser[_p_ser.StudyInstanceUID.isin(_p_ids)].to_csv(_P_FAKE / "test_series.csv", index=False)
_p_sub = pd.DataFrame({"StudyInstanceUID": _p_ids})
for _t in _P_T:
    _p_sub[_t] = 0.5
_p_sub.to_csv(_P_FAKE / "sample_submission.csv", index=False)
if not (_P_FAKE / "train.csv").exists():
    os.symlink(_P_REAL / "train.csv", _P_FAKE / "train.csv")           # the pipeline reads its row count
os.environ["RSNA_COMP_ROOT"] = str(_P_FAKE)
print(f"probe root {_P_FAKE}: {len(_p_ids)} studies, {int(_p_ser.StudyInstanceUID.isin(_p_ids).sum())} series", flush=True)
