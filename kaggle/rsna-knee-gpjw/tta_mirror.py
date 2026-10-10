"""Does an anatomical-mirror second view help the 2.5D reader? (check run, inference only)

Mirror = the left-right mirror image of the patient, applied consistently to all planes, so a right
knee becomes a valid left knee and every finding keeps its label (medial stays medial):
  coronal and axial series: each slice flipped left-right;
  sagittal series: slice order reversed, i.e. position p -> 1 - p and the 3-slice window's channel order flipped.
Each fold model scores its own held-out fold and the gold studies twice, plain and mirrored.

Usage: python tta_mirror.py <shard index json> <meta csv> <series csv> <folds csv> <out dir> <fold>=<checkpoint> ...
"""
import sys, time

import pandas as pd
import torch

from infer import load_models
from knee import LABELS, StudyDataset, build_study_table
from shards import ShardLoader

index_json, meta_csv, series_csv, folds_csv, out_dir = sys.argv[1:6]
ckpts = dict(a.split("=", 1) for a in sys.argv[6:])
SAG, OTHER = [0, 3], [1, 2, 4, 5]          # slot = plane (sagittal, coronal, axial) + 3 * (not fat-suppressed)


def mirror(x, pos):
    x, pos = x.clone(), pos.clone()
    x[:, SAG] = x[:, SAG].flip(3)
    pos[:, SAG] = 1 - pos[:, SAG]
    x[:, OTHER] = x[:, OTHER].flip(-1)
    return x, pos


lab = pd.read_csv(folds_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
meta = pd.read_csv(meta_csv, dtype={"study": str, "series": str})
series = pd.read_csv(series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0))))
loader = ShardLoader(index_json)
gold = lab.index[lab.is_gold == 1].tolist()
for fold, ck in sorted(ckpts.items()):
    t0 = time.time()
    val = lab.index[lab.fold == int(fold)].tolist()
    ids = val + gold
    (model, res), = load_models([ck])
    dl = torch.utils.data.DataLoader(StudyDataset(ids, table, None, None, 12, loader=loader), batch_size=4, num_workers=4)
    out = {"plain": [], "mirror": []}
    with torch.no_grad():
        for x, mask, pos, _ in dl:
            x, mask, pos = x.cuda(), mask.cuda(), pos.cuda()
            xm, pm = mirror(x, pos)
            with torch.autocast("cuda", dtype=torch.float16):
                out["plain"].append(model(x, mask, pos, res=res).float().sigmoid().cpu())
                out["mirror"].append(model(xm, mask, pm, res=res).float().sigmoid().cpu())
    parts = []
    for view, o in out.items():
        df = pd.DataFrame(torch.cat(o).numpy(), index=pd.Index(ids, name="StudyInstanceUID"), columns=LABELS)
        df.insert(0, "view", view)
        df.insert(0, "set", ["val"] * len(val) + ["gold"] * len(gold))
        parts.append(df)
    pd.concat(parts).to_csv(f"{out_dir}/tta_fold{fold}.csv")
    print(f"fold {fold} ({ck.rsplit('/', 1)[-1]}): {len(val)} fold studies + {len(gold)} gold, two views, {time.time() - t0:.0f}s", flush=True)
    del model
    torch.cuda.empty_cache()
