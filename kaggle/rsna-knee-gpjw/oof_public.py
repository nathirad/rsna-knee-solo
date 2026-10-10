"""Out-of-fold predictions of the public fold checkpoints: each checkpoint scores the studies of its
own fold (the ones it did not train on) and the gold studies, through our cache. With our own
folds 3 and 4 this gives an out-of-fold prediction for every train study: the input of the next
label round (labels mixed with out-of-fold predictions).

Usage: python oof_public.py <checkpoint dir> <shard index json> <meta csv> <series csv> <labels csv> <out dir>
"""
import os, sys, time

import pandas as pd
import torch

from infer import load_models
from knee import LABELS, StudyDataset, build_study_table
from shards import ShardLoader

ck_dir, index_json, meta_csv, series_csv, labels_csv, out_dir = sys.argv[1:7]
lab = pd.read_csv(labels_csv, index_col=0)
lab.index = lab.index.astype(str)
meta = pd.read_csv(meta_csv, dtype={"study": str, "series": str})
series = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0))))
loader = ShardLoader(index_json)
gold = lab.index[lab.is_gold == 1].tolist()
for k in range(5):
    ck = f"{ck_dir}/cnxt_v0_fold{k}.pt"
    if not os.path.exists(ck):
        continue
    t0 = time.time()
    val = lab.index[lab.fold == k].tolist()
    (model, res), = load_models([ck])
    dl = torch.utils.data.DataLoader(StudyDataset(val + gold, table, None, None, 12, loader=loader), batch_size=4, num_workers=4)
    out = []
    with torch.no_grad():
        for x, mask, pos, _ in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                out.append(model(x.cuda(), mask.cuda(), pos.cuda(), res=res).float().sigmoid().cpu())
    df = pd.DataFrame(torch.cat(out).numpy(), index=pd.Index(val + gold, name="StudyInstanceUID"), columns=LABELS)
    df.insert(0, "set", ["val"] * len(val) + ["gold"] * len(gold))
    df.to_csv(f"{out_dir}/oof_public_fold{k}.csv")
    print(f"public fold {k}: {len(val)} fold studies + {len(gold)} gold in {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
