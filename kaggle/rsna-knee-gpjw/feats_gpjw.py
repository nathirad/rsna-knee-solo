"""Backbone features of one fold model for every study, so that its head can be retrained on a CPU.

The reader is backbone -> head (`KneeNet.head`: projection, slot and position embeddings, a 2-layer
transformer, one attention pooling per finding). The backbone is nearly all of the cost. With its
per-window features saved once, a head can be retrained under another label table in minutes, which
is how label policies are compared before a full training run is paid for.

Windows are the evaluation ones (--k_eval evenly spaced 3-slice windows per series, no augmentation),
so `head(feat)` here is the checkpoint's own prediction: `logit` is saved for that check.
Order: gold studies, the held-out fold, then the training folds; --budget-h stops early and saves
what is done.

Written to <out>/feats_<tag>_fold<k>.npz: studies, feat float16 [N, 6, K, D], mask [N, 6],
pos [N, 6, K], logit float32 [N, 12].

Usage: python feats_gpjw.py --tag ctrl --fold 3 --init ctrl_fold3.pt --index JSON --labels CSV --meta CSV --series-csv CSV --out DIR
"""
import argparse, json, os, time

import numpy as np
import pandas as pd
import torch

from knee import KneeNet, StudyDataset, build_study_table, make_views
from shards import ShardLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--init", required=True, help="checkpoint (public format) whose backbone is run")
    ap.add_argument("--index", required=True, help="json: series -> shard file (written by the session cell)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--k_eval", type=int, default=12)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--budget-h", type=float, default=0, help="hours this job may take; 0 = no limit")
    ap.add_argument("--limit", type=int, default=0, help="debug: only this many studies")
    args = ap.parse_args()
    t_start = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.tag}_fold{args.fold}"
    print(json.dumps(vars(args)), flush=True)

    lab = pd.read_csv(args.labels, index_col=0)
    lab.index = lab.index.astype(str)
    meta = pd.read_csv(args.meta, dtype={"study": str, "series": str})
    series = pd.read_csv(args.series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0))))
    weak = lab[lab.is_gold == 0]
    order = list(lab.index[lab.is_gold == 1]) + list(weak.index[weak.fold == args.fold]) + list(weak.index[weak.fold != args.fold])
    if args.limit:
        order = order[:args.limit]

    model = KneeNet(args.backbone, pretrained=False).to(dev).to(memory_format=torch.channels_last).eval()
    model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
    print(f"backbone and head from {os.path.basename(args.init)}; {len(order)} studies", flush=True)
    dl = torch.utils.data.DataLoader(StudyDataset(order, table, None, None, args.k_eval, loader=ShardLoader(args.index)),
                                     batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=amp)
    feats, masks, poss, logits, n = [], [], [], [], 0
    with torch.no_grad():
        for x, mask, pos, _ in dl:
            x, mask, pos = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev)
            b, s, k = x.shape[:3]
            keep = mask.view(-1)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                feat = torch.zeros(b * s, k, model.enc.num_features, device=dev)
                if keep.any():                       # as KneeNet.forward: the backbone runs only on series that exist
                    xv = x.view(b * s, k, *x.shape[3:])[keep].flatten(0, 1)
                    f = model.enc(make_views(xv, args.res, False, k).contiguous(memory_format=torch.channels_last))
                    feat = feat.to(f.dtype)
                    feat[keep] = f.view(-1, k, f.shape[-1])
                feat = feat.view(b, s, k, -1)
                logit = model.head(feat, mask, pos)
            feats.append(feat.to(torch.float16).cpu().numpy())
            masks.append(mask.cpu().numpy())
            poss.append(pos.cpu().numpy())
            logits.append(logit.float().cpu().numpy())
            n += b
            if (n // b) % 100 == 0:
                print(f"  {n}/{len(order)} studies, {(time.time() - t_start) / max(n, 1):.2f} s/study", flush=True)
            if args.budget_h and time.time() - t_start > args.budget_h * 3600:
                print(f"budget of {args.budget_h} h reached at {n} studies: the rest is not saved", flush=True)
                break
    np.savez(f"{args.out}/feats_{name}.npz", studies=np.array(order[:n]), feat=np.concatenate(feats), mask=np.concatenate(masks),
             pos=np.concatenate(poss), logit=np.concatenate(logits))
    print(f"saved feats_{name}.npz: {n} of {len(order)} studies, feat {np.concatenate(feats).shape}, "
          f"{(time.time() - t_start) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
