"""Retrain the head of one fold model on its saved backbone features, and write a full checkpoint.

The reader is backbone -> head (`KneeNet.head`). `feats_gpjw.py` saved the backbone's features for
every study, so a head can be trained under another label table on a CPU in a few minutes. This
script does that for one fold and writes the result in the public checkpoint format (the backbone of
--init with the new head), so the reader's `infer.py` loads it like any other checkpoint.

Training follows the reader's recipe for the head (AdamW 5e-4, weight decay 0.02, cosine schedule with
a one-epoch warm-up, gradient clipping, EMA 0.998, 8 of the 12 windows of every series per step) for
--epochs epochs: on frozen features the head is at its best after about four (research/head_lab.py,
where the label policies were compared), then it overfits.

Written to <out>: <name>_fold<k>.pt, oof_<name>_fold<k>.csv (probabilities on the gold studies and
the held-out fold, as the training job writes them), results_<name>_fold<k>.json.

Usage: python fit_heads.py --feats feats_ctrl_fold3.npz --labels labels.csv --init ctrl_fold3.pt --name calFG --out DIR
"""
import argparse, copy, json, math, os, re, time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
          "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


class Head(nn.Module):
    """KneeNet.head with the same parameter names, so its weights go straight into a checkpoint."""

    def __init__(self, d_in, dim=384, depth=2, drop=0.2, n_slots=6):
        super().__init__()
        self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, dim), nn.LayerNorm(dim))
        self.slot_emb = nn.Embedding(n_slots, dim)
        self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(LABELS)))
        self.cls_w = nn.Parameter(torch.randn(len(LABELS), dim) * 0.02)
        self.cls_b = nn.Parameter(torch.zeros(len(LABELS)))

    def forward(self, feat, mask, pos):
        b, s, k, _ = feat.shape
        slot = torch.arange(s).view(1, s, 1).expand(b, s, k)
        h = self.proj(feat) + self.slot_emb(slot) + self.pos_emb(pos.unsqueeze(-1))
        h = h.view(b, s * k, -1)
        pad = ~mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        h = self.tf(h, src_key_padding_mask=pad)
        att = self.att(h).masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        return (torch.einsum("btc,btd->bcd", att, h) * self.cls_w).sum(-1) + self.cls_b


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), [float(a) for a in aucs]


def train_head(feat, mask, pos, tgt, wgt, tr, seed, epochs=4, bs=8, k=8, lr=5e-4, wd=0.02, ema_decay=0.998):
    """One head on the training rows `tr`; returns the EMA weights' module (eval mode)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Head(feat.shape[-1])
    ema = copy.deepcopy(net).eval()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    spe = len(tr) // bs
    steps = epochs * spe
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    step = 0
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr))
        for i in range(spe):
            r = np.sort(tr[order[i * bs:(i + 1) * bs]])
            f, p = torch.from_numpy(feat[r].astype(np.float32)), torch.from_numpy(pos[r])
            # as in training: a random subset of the windows of every series
            idx = torch.rand(f.shape[:3]).argsort(-1)[..., :k].sort(-1).values
            f = f.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, f.shape[-1]))
            logit = net(f, torch.from_numpy(mask[r]), p.gather(2, idx))
            loss = (F.binary_cross_entropy_with_logits(logit, torch.from_numpy(tgt[r]), reduction="none") * torch.from_numpy(wgt[r])).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            sched.step()
            step += 1
            d = min(ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), net.parameters()):
                    pe.lerp_(pm, 1 - d)
    return ema


def predict(net, feat, mask, pos, rows):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            r = rows[i:i + 64]
            out.append(net(torch.from_numpy(feat[r].astype(np.float32)), torch.from_numpy(mask[r]), torch.from_numpy(pos[r])).sigmoid())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--labels", required=True, help="table with targets, optional w_<finding> weights, fold, is_gold, gold_<finding>")
    ap.add_argument("--init", required=True, help="checkpoint whose backbone made the features")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    fold = int(re.search(r"fold(\d+)", os.path.basename(a.feats)).group(1))
    name = f"{a.name}_fold{fold}"
    z = np.load(a.feats)
    studies = [str(s) for s in z["studies"]]
    feat, mask, pos = z["feat"], z["mask"], z["pos"].astype(np.float32)
    lab = pd.read_csv(a.labels, index_col=0)
    lab.index = lab.index.astype(str)
    lab = lab.loc[studies]
    is_gold, fo = (lab.is_gold == 1).values, lab.fold.values
    tr, va, go = np.where(~is_gold & (fo != fold))[0], np.where(~is_gold & (fo == fold))[0], np.where(is_gold)[0]
    assert len(tr) > 3000 or os.environ.get("GPJW_LOCAL"), f"only {len(tr)} training studies have features"
    tgt = lab[LABELS].values.astype(np.float32)
    w_cols = [f"w_{c}" for c in LABELS]
    wgt = lab[w_cols].values.astype(np.float32) if w_cols[0] in lab.columns else np.ones_like(tgt)
    gold_y = lab.iloc[go][[f"gold_{c}" for c in LABELS]].values.astype(int)
    val_cols = [f"orig_{c}" for c in LABELS] if f"orig_{LABELS[0]}" in lab.columns else LABELS
    val_y = (lab.iloc[va][val_cols].values > 0.5).astype(int)
    before = 1 / (1 + np.exp(-z["logit"]))
    g0, g0_each = macro_auc(gold_y, before[go])
    print(f"fold {fold}: train {len(tr)}, held-out fold {len(va)}, gold {len(go)}; the checkpoint's own head: gold {g0:.4f}, "
          f"held-out fold {macro_auc(val_y, before[va])[0]:.4f}", flush=True)

    head = train_head(feat, mask, pos, tgt, wgt, tr, a.seed, a.epochs)
    ev = np.concatenate([go, va])
    p = predict(head, feat, mask, pos, ev)
    g1, g1_each = macro_auc(gold_y, p[:len(go)])
    v1, v1_each = macro_auc(val_y, p[len(go):])
    print(f"new head ({a.name}, seed {a.seed}, {a.epochs} epochs): gold {g1:.4f} ({g1 - g0:+.4f}), held-out fold {v1:.4f}", flush=True)
    print("gold per class, before -> after: " + " ".join(f"{c[:6]}={b:.3f}->{n:.3f}" for c, b, n in zip(LABELS, g0_each, g1_each)), flush=True)

    # ---- the backbone of --init with the new head, in the public checkpoint format ----
    from knee import KneeNet
    ck = torch.load(a.init, map_location="cpu", weights_only=False)
    sd, new = ck["model"], head.state_dict()
    assert set(new) <= set(sd) and all(sd[k].shape == v.shape for k, v in new.items()), "the head does not fit this checkpoint"
    assert {k for k in sd if not k.startswith("enc.")} == set(new), "the checkpoint has head weights this script does not train"
    sd.update({k: v.clone() for k, v in new.items()})
    model = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        r = go[:16]
        again = model.head(torch.from_numpy(feat[r].astype(np.float32)), torch.from_numpy(mask[r]), torch.from_numpy(pos[r])).sigmoid().numpy()
    diff = float(np.abs(again - p[:len(r)]).max())
    assert diff < 1e-4, f"the merged checkpoint's head differs from the trained head: {diff}"
    ck["args"] = {**ck["args"], "head": {"labels": a.name, "seed": a.seed, "epochs": a.epochs, "backbone_of": os.path.basename(a.init)}}
    torch.save({"model": sd, "args": ck["args"]}, f"{a.out}/{name}.pt")
    df = pd.DataFrame(p, index=lab.index[ev], columns=LABELS)
    df.insert(0, "set", ["gold"] * len(go) + ["val"] * len(va))
    df.to_csv(f"{a.out}/oof_{name}.csv")
    json.dump({"name": a.name, "fold": fold, "init": os.path.basename(a.init), "seed": a.seed, "epochs": a.epochs,
               "gold_before": g0, "gold": g1, "gold_each_before": dict(zip(LABELS, g0_each)), "gold_each": dict(zip(LABELS, g1_each)),
               "val": v1, "val_each": dict(zip(LABELS, v1_each)), "merged_head_max_abs_diff": diff, "minutes": (time.time() - t0) / 60},
              open(f"{a.out}/results_{name}.json", "w"), indent=1)
    print(f"saved {name}.pt (backbone of {os.path.basename(a.init)}, new head; merged head check {diff:.1e}) in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
