"""Head-only label experiments: retrain the reader's head on saved backbone features, on a CPU.

The reader is backbone -> head (kaggle/rsna-knee-gpjw `KneeNet.head`). `feats_gpjw.py` saved the
backbone's per-window features of one fold model for every study, so a new head can be trained under
any label policy in about two minutes. The backbone was trained on the folds other than its own, so
its own fold and the 58 expert-labelled studies are held out for every head trained here.

  run      train heads for a list of label policies on one feature file and save their predictions
  compare  gold-58 table of the saved predictions (per fold backbone and both together), with a
           paired bootstrap of each policy against the first one

Policies, written target[:weight]:
  raw        the four-table consensus (what the reader was trained on)            weight one
  polA       pilkwang graded scores for Effusion and Synovitis (the fine-tune of 2026-10-09)
  pilk       pilkwang graded scores for all findings (the older solo model's labels)
  pooled / finding / grades    the consensus or the pilkwang grade mapped to P(expert positive)
             (research/label_calibration.py), fitted on the gold studies named by --fit-gold
  a+b        the mean of the targets of policies a and b, e.g. raw+grades
  weight: one | conf (0.3 + 0.7*|2*target - 1|) | pconf (0.25 + 0.75*pilkwang confidence)
--fit-gold A or B fits the mapping on one half of the gold studies; `compare` then scores each half's
heads on the other half only (the honest number for a mapping that was itself fitted on gold).
--gold-weight W (with --fit-gold A or B) also trains on that half's studies with their expert labels,
each repeated W times; saved as policy@gW. Only the other half counts, as above. (Pooling the held-out
predictions of a K-fold split over the 58 was tried first and is not used: heads that saw different
gold studies sit on different scales, which lowers the pooled AUC by itself.)

Usage (from the RSNA project folder):
  python research/head_lab.py run --feats runs/rsna-knee-gpjw-feats-a/latest/res/feats_ctrl_fold3.npz --policies raw finding grades:conf
  python research/head_lab.py compare --folds 3 4
"""
import argparse, copy, math, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).parent))
from label_calibration import Calibrator, T, load          # noqa: E402

OUT = Path("runs/head_lab")
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


def auc(y, p):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per(y, p, rows=None):
    if rows is not None:
        y, p = y[rows], p[rows]
    return np.array([auc(y[:, j], p[:, j]) for j in range(y.shape[1])])


def gold_halves(gold_ids):
    """A fixed split of the gold studies into two halves (for mappings that are fitted on gold)."""
    perm = np.random.RandomState(58).permutation(len(gold_ids))
    half = np.zeros(len(gold_ids), bool)
    half[perm[:len(gold_ids) // 2]] = True
    return {"A": half, "B": ~half, "all": np.ones(len(gold_ids), bool)}


def targets_for(policy, lab, tables, fit_rows):
    """Targets and loss weights [n studies, 12] for every row of `lab`, for one policy string."""
    cons, lev, src = tables
    name, _, wname = policy.partition(":")
    g = np.where(lab.is_gold == 1)[0][fit_rows]
    y_gold = lab.iloc[g][[f"gold_{c}" for c in T]].values.astype(float)
    w = np.ones_like(cons)
    if "+" in name:                                  # a target that is the mean of several policies' targets
        t = np.mean([targets_for(part, lab, tables, fit_rows)[0] for part in name.split("+")], 0)
    elif name == "raw":
        t = cons.copy()
    elif name == "polA":
        pa = ld("runs/solo/gpjw_labels_polA.csv").reindex(lab.index)
        t, w = pa[T].values, pa[[f"w_{c}" for c in T]].values
    elif name == "pilk":
        pk = ld("research/public_notebooks/knee_g2/lab1/report_labels_v2.csv").reindex(lab.index)
        t = pk[T].astype(float).fillna(lab[T]).clip(0.01, 0.99).values
    else:
        t = Calibrator(name).fit(cons[g], lev[g], src[g], y_gold).predict(cons, lev, src)
    if wname == "conf":
        w = 0.3 + 0.7 * np.abs(2 * t - 1)
    elif wname == "pconf":
        pk = ld("research/public_notebooks/knee_g2/lab1/report_labels_v2.csv").reindex(lab.index)
        w = 0.25 + 0.75 * pk[[f"{c}__conf" for c in T]].astype(float).fillna(0.5).values
    else:
        assert wname in ("", "one"), f"unknown weight {wname!r}"
    return t.astype(np.float32), w.astype(np.float32)


def run(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        """KneeNet.head with the same parameter names, so its weights can go back into a checkpoint."""

        def __init__(self, d_in, dim=384, depth=2, drop=0.2, n_slots=6):
            super().__init__()
            self.proj = nn.Sequential(nn.Dropout(drop), nn.Linear(d_in, dim), nn.LayerNorm(dim))
            self.slot_emb = nn.Embedding(n_slots, dim)
            self.pos_emb = nn.Sequential(nn.Linear(1, dim), nn.GELU(), nn.Linear(dim, dim))
            layer = nn.TransformerEncoderLayer(dim, 8, dim * 2, dropout=0.1, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.att = nn.Sequential(nn.Linear(dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, len(T)))
            self.cls_w = nn.Parameter(torch.randn(len(T), dim) * 0.02)
            self.cls_b = nn.Parameter(torch.zeros(len(T)))

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

    torch.set_num_threads(a.threads)
    fold = int(re.search(r"fold(\d+)", Path(a.feats).name).group(1))
    z = np.load(a.feats)
    studies = [str(s) for s in z["studies"]]
    feat, mask, pos = z["feat"], z["mask"], z["pos"].astype(np.float32)
    lab, cons, lev, src = load()
    row = pd.Series(np.arange(len(lab)), index=lab.index).reindex(studies).values
    assert not np.isnan(row).any(), "a study of the feature file is not in the label table"
    lab, tables = lab.iloc[row], (cons[row], lev[row], src[row])
    is_gold, fo = (lab.is_gold == 1).values, lab.fold.values
    tr, va, go = np.where(~is_gold & (fo != fold))[0], np.where(~is_gold & (fo == fold))[0], np.where(is_gold)[0]
    gold_y = lab.iloc[go][[f"gold_{c}" for c in T]].values.astype(int)
    val_y = (tables[0][va] > 0.5).astype(int)
    halves = gold_halves(lab.index[go])
    OUT.mkdir(parents=True, exist_ok=True)
    ev = np.concatenate([go, va])
    sets = np.array(["gold"] * len(go) + ["val"] * len(va))
    base = 1 / (1 + np.exp(-z["logit"][ev]))
    pd.DataFrame(base, index=lab.index[ev], columns=T).assign(set=sets).to_csv(OUT / f"pred_checkpoint_fold{fold}.csv")
    print(f"fold {fold}: train {len(tr)}, held-out fold {len(va)}, gold {len(go)}; the checkpoint itself: gold {per(gold_y, base[:len(go)]).mean():.4f}, "
          f"held-out fold vs consensus {per(val_y, base[len(go):]).mean():.4f}", flush=True)

    def predict(net, rows):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 64):
                r = rows[i:i + 64]
                out.append(net(torch.from_numpy(feat[r].astype(np.float32)), torch.from_numpy(mask[r]), torch.from_numpy(pos[r])).sigmoid())
        return torch.cat(out).numpy()

    def fit(tgt, wgt, tr, seed):
        """One head (EMA weights) trained on the rows `tr`; the per-epoch trace when asked for."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Head(feat.shape[-1])
        ema = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        spe = len(tr) // a.bs
        steps = a.epochs * spe
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        step, trace = 0, []
        for ep in range(a.epochs):
            net.train()
            order = rng.permutation(len(tr))
            for i in range(spe):
                r = np.sort(tr[order[i * a.bs:(i + 1) * a.bs]])
                f, p = torch.from_numpy(feat[r].astype(np.float32)), torch.from_numpy(pos[r])
                # as in training: a random subset of the windows of every series
                idx = torch.rand(f.shape[:3]).argsort(-1)[..., :a.k].sort(-1).values
                f = f.gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, f.shape[-1]))
                logit = net(f, torch.from_numpy(mask[r]), p.gather(2, idx))
                loss = (F.binary_cross_entropy_with_logits(logit, torch.from_numpy(tgt[r]), reduction="none") * torch.from_numpy(wgt[r])).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
                opt.step()
                sched.step()
                step += 1
                d = min(a.ema, (1 + step) / (10 + step))
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), net.parameters()):
                        pe.lerp_(pm, 1 - d)
            if a.trace and (ep % 2 == 1 or ep == a.epochs - 1):
                trace.append(f"ep{ep + 1} gold {per(gold_y, predict(ema, go)).mean():.4f} val {per(val_y, predict(ema, va)).mean():.4f}")
        return ema, trace

    assert not a.gold_weight or a.fit_gold in ("A", "B"), "--gold-weight needs --fit-gold A or B: the other half is the test"
    tag = a.fit_gold
    for policy in a.policies:
        t0 = time.time()
        tgt, wgt = targets_for(policy, lab, tables, halves[a.fit_gold])
        rows = tr
        if a.gold_weight:                              # this half's gold studies join the training set with their expert labels
            tgt[go] = np.clip(gold_y, 0.02, 0.98)
            rows = np.concatenate([tr] + [go[halves[a.fit_gold]]] * a.gold_weight)
            policy = f"{policy}@g{a.gold_weight}"
        preds = []
        for seed in a.seeds:
            ema, trace = fit(tgt, wgt, rows, seed)
            p = predict(ema, ev)
            preds.append(p)
            print(f"  {policy:16s} seed {seed}: gold {per(gold_y, p[:len(go)]).mean():.4f}  held-out fold vs consensus {per(val_y, p[len(go):]).mean():.4f}"
                  + ("  | " + ", ".join(trace) if trace else ""), flush=True)
            if a.save_heads:
                torch.save(ema.state_dict(), OUT / f"head_{policy.replace(':', '-')}_{tag}_fold{fold}_s{seed}.pt")
        mean = np.mean(preds, 0)
        pd.DataFrame(mean, index=lab.index[ev], columns=T).assign(set=sets).to_csv(OUT / f"pred_{policy.replace(':', '-')}_{tag}_fold{fold}.csv")
        print(f"{policy:16s} fit-gold {tag}: {len(a.seeds)} seeds together: gold {per(gold_y, mean[:len(go)]).mean():.4f}, "
              f"held-out fold vs consensus {per(val_y, mean[len(go):]).mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)


def compare(a):
    lab = ld("runs/solo/gpjw_labels_v0.csv")
    gid = lab.index[lab.is_gold == 1]
    y = lab.loc[gid, [f"gold_{c}" for c in T]].values.astype(int)
    halves = gold_halves(gid)
    names = sorted({re.sub(r"_fold\d+\.csv$", "", f.name[5:]) for f in OUT.glob("pred_*_fold*.csv")})
    names = [n for n in a.first if n in names] + [n for n in names if n not in a.first]
    rng = np.random.default_rng(0)
    rows, P = [], {}
    for n in names:
        files = [OUT / f"pred_{n}_fold{k}.csv" for k in a.folds]
        if not all(f.exists() for f in files):
            continue
        ps = [(lambda d: d[d.set == "gold"].loc[gid, T].values)(ld(f)) for f in files]
        P[n] = np.mean(ps, 0)
        r = {"policy": n, **{f"fold {k}": per(y, p).mean() for k, p in zip(a.folds, ps)}, "together": per(y, P[n]).mean()}
        fit = n.rsplit("_", 1)[-1]
        if fit in ("A", "B"):                     # heads whose mapping saw half of gold: only the other half counts
            r["other half"] = np.nanmean(per(y, P[n], np.where(~halves[fit])[0]))
        rows.append(r)
    tab = pd.DataFrame(rows).set_index("policy")
    ref = tab.index[0]
    boots = [rng.integers(0, len(y), len(y)) for _ in range(2000)]
    for n in tab.index[1:]:
        d = np.array([np.nanmean(per(y, P[n], b)) - np.nanmean(per(y, P[ref], b)) for b in boots])
        tab.loc[n, f"vs {ref}"] = tab.loc[n, "together"] - tab.loc[ref, "together"]
        tab.loc[n, "CI low"], tab.loc[n, "CI high"], tab.loc[n, "P>0"] = np.percentile(d, 2.5), np.percentile(d, 97.5), (d > 0).mean()
    print(f"gold-58 macro AUC of retrained heads (folds {a.folds}; 'together' = mean of the fold models' probabilities)")
    print(tab.round(4).to_string())
    print("\nper finding, the fold models together")
    print(pd.DataFrame({n: per(y, P[n]) for n in tab.index}, index=T).round(3).T.to_string())
    # cross-fitted pairs: A-heads scored on half B and B-heads on half A, pooled into one honest number per policy
    pairs = sorted({n[:-2] for n in P if n.endswith(("_A", "_B"))})
    for base in pairs:
        if f"{base}_A" in P and f"{base}_B" in P:
            sc = np.nanmean([np.nanmean(per(y, P[f"{base}_A"], np.where(halves["B"])[0])), np.nanmean(per(y, P[f"{base}_B"], np.where(halves["A"])[0]))])
            rf = np.nanmean([np.nanmean(per(y, P[ref], np.where(halves["B"])[0])), np.nanmean(per(y, P[ref], np.where(halves["A"])[0]))])
            print(f"cross-fitted {base}: mean of the two half-scores {sc:.4f} vs {ref} on the same halves {rf:.4f} ({sc - rf:+.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--feats", required=True)
    r.add_argument("--policies", nargs="+", required=True)
    r.add_argument("--fit-gold", default="all", choices=["all", "A", "B"])
    r.add_argument("--gold-weight", type=int, default=0, help="also train on the --fit-gold half with expert labels, each study this many times")
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--epochs", type=int, default=12)
    r.add_argument("--bs", type=int, default=8)
    r.add_argument("--k", type=int, default=8)
    r.add_argument("--lr", type=float, default=5e-4)
    r.add_argument("--wd", type=float, default=0.02)
    r.add_argument("--ema", type=float, default=0.998)
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--trace", action="store_true", help="print gold and held-out AUC every two epochs")
    r.add_argument("--save-heads", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--folds", nargs="+", type=int, default=[3, 4])
    c.add_argument("--first", nargs="*", default=["checkpoint", "raw_all"], help="policies listed first; the first one present is the reference")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
