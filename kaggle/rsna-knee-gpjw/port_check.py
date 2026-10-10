"""Check this cache and loader against the public pipeline before any training time is spent:
the public fold checkpoint, run on the gold studies through this cache, must give the predictions
it gave through the reader's own DICOM path (rsna-knee-gpjw-check).

Usage: python port_check.py <checkpoint> <shard index json> <meta csv> <series csv> <reference json> <out json>
"""
import json, os, sys

import numpy as np
import pandas as pd
import torch

from infer import load_models
from knee import StudyDataset, build_study_table
from shards import ShardLoader

ck_path, index_json, meta_csv, series_csv, ref_json, out_json = sys.argv[1:7]
ref = json.load(open(ref_json))
meta = pd.read_csv(meta_csv, dtype={"study": str, "series": str})
series = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0))))
args = torch.load(ck_path, map_location="cpu", weights_only=False)["args"]
(model, res), = load_models([ck_path])
dl = torch.utils.data.DataLoader(StudyDataset(ref["studies"], table, None, None, 12, loader=ShardLoader(index_json)),
                                 batch_size=4, num_workers=int(os.environ.get("GPJW_PORT_WORKERS", "0")))   # 58 studies: no workers needed
out = []
with torch.no_grad():
    for x, mask, pos, _ in dl:
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(x.cuda(), mask.cuda(), pos.cuda(), res=res).float().sigmoid().cpu())
d = np.abs(torch.cat(out).numpy() - np.array(ref["pred"])).max(1)
result = {"checkpoint": ref["checkpoint"], "checkpoint_args": args, "studies": len(d),
          "max_abs_diff": float(d.max()), "median_abs_diff": float(np.median(d)),
          "studies_over_0.02": int((d > 0.02).sum()), "ok": bool((d > 0.02).mean() <= 0.10)}
json.dump(result, open(out_json, "w"), indent=1, default=str)
print(json.dumps(result, indent=1, default=str), flush=True)
sys.exit(0 if result["ok"] else 1)
