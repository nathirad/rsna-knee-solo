"""Head-only label experiments on the public single CoAtNet ("Raptor" family), on a CPU.

The model is backbone -> head, and the head is small: LayerNorm, one attention pooling per finding over
the 94 windows, one linear score per finding (RaptorClassifier.head in the public notebook). build_feats.py
saved the backbone's features of every train study, so heads can be retrained here in seconds under any
label policy. The backbone was trained on all 4,349 report-labelled studies; the 58 expert-labelled ones
were held out by its author (and reused for checkpoint selection), so they are the test here, and one of
our folds of the train studies is kept aside only to watch for overfitting to the report labels.

  run      train heads for a list of label policies (research/head_lab.py's policy strings: raw, grades,
           finding+grades, ...; `ft:` in front starts from the checkpoint's own head, needs --head)
  compare  gold-58 table of the saved predictions against the checkpoint's own head, with a paired bootstrap;
           policies fitted on half of gold (--fit-gold A/B) are scored on the other half

Usage (from the RSNA project folder):
  python kaggle/rsna-knee-raptor/head_lab.py run --feats runs/rsna-knee-raptor224-feats/latest/res/feats_raptor_train.npz --tag b224 --policies raw finding+grades
  python kaggle/rsna-knee-raptor/head_lab.py compare --tag b224
"""
import argparse, copy, math, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

RSNA = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RSNA / "research"))
from head_lab import gold_halves, per, targets_for            # noqa: E402
from label_calibration import T, load                          # noqa: E402

OUT = Path("runs/raptor_head_lab")
ld = lambda p: pd.read_csv(p, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")


def run(a):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        """RaptorClassifier.head with the same parameter names (norm, att, clsW, clsb)."""

        def __init__(self, f_dim, n=len(T), drop=0.2):
            super().__init__()
            self.norm = nn.LayerNorm(f_dim)
            self.att = nn.Sequential(nn.Linear(f_dim, 256), nn.Tanh(), nn.Dropout(drop), nn.Linear(256, n))
            self.clsW = nn.Parameter(torch.zeros(n, f_dim))
            self.clsb = nn.Parameter(torch.zeros(n))
            nn.init.trunc_normal_(self.clsW, std=0.02)

        def forward(self, feats):
            h = self.norm(feats)
            att = torch.softmax(self.att(h), dim=1)
            return (torch.einsum("bkn,bkf->bnf", att, h) * self.clsW).sum(-1) + self.clsb

    torch.set_num_threads(a.threads)
    z = np.load(a.feats)
    studies = [str(s) for s in z["studies"]]
    feat = z["feat"]
    lab, cons, lev, src = load()
    row = pd.Series(np.arange(len(lab)), index=lab.index).reindex(studies).values
    assert not np.isnan(row).any(), "a study of the feature file is not in the label table"
    lab, tables = lab.iloc[row], (cons[row], lev[row], src[row])
    is_gold, fo = (lab.is_gold == 1).values, lab.fold.values
    go, va = np.where(is_gold)[0], np.where(~is_gold & (fo == a.watch_fold))[0]
    tr = np.where(~is_gold & (fo != a.watch_fold))[0] if not a.all_train else np.where(~is_gold)[0]
    gold_y = lab.iloc[go][[f"gold_{c}" for c in T]].values.astype(int)
    val_y = (tables[0][va] > 0.5).astype(int)
    halves = gold_halves(lab.index[go])
    OUT.mkdir(parents=True, exist_ok=True)
    ev = np.concatenate([go, va])
    sets = np.array(["gold"] * len(go) + ["val"] * len(va))
    base = 1 / (1 + np.exp(-z["logit"][ev]))
    pd.DataFrame(base, index=lab.index[ev], columns=T).assign(set=sets).to_csv(OUT / f"pred_{a.tag}_checkpoint.csv")
    print(f"{a.tag}: {len(studies)} studies, feat {feat.shape}; train {len(tr)}, watch fold {len(va)}, gold {len(go)}; the checkpoint's own head: "
          f"gold {per(gold_y, base[:len(go)]).mean():.4f}, watch fold vs consensus {per(val_y, base[len(go):]).mean():.4f} (its backbone trained on these)", flush=True)
    own = None
    if a.head:                                                     # the checkpoint's head weights (export_heads kernel)
        own = {k: torch.from_numpy(v) for k, v in np.load(a.head).items()}
        chk = Head(feat.shape[-1])
        chk.load_state_dict(own)
        chk.eval()
        with torch.no_grad():
            d = float((chk(torch.from_numpy(feat[go].astype(np.float32))).numpy() - z["logit"][go]).__abs__().max())
        assert d < 5e-3, f"the exported head does not reproduce the saved logits: {d}"
        print(f"exported head reproduces the saved logits on gold (max abs diff {d:.1e})", flush=True)

    def predict(net, rows):
        net.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 128):
                out.append(net(torch.from_numpy(feat[rows[i:i + 128]].astype(np.float32))).sigmoid())
        return torch.cat(out).numpy()

    def fit(tgt, wgt, rows, seed, init):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = Head(feat.shape[-1])
        if init is not None:
            net.load_state_dict(init)
        ema = copy.deepcopy(net).eval()
        lr = a.lr * (0.2 if init is not None else 1.0)             # a head that starts from the checkpoint's moves gently
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=a.wd)
        spe = len(rows) // a.bs
        steps = a.epochs * spe
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / spe) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
        step, trace = 0, []
        for ep in range(a.epochs):
            net.train()
            order = rng.permutation(len(rows))
            for i in range(spe):
                r = np.sort(rows[order[i * a.bs:(i + 1) * a.bs]])
                f = torch.from_numpy(feat[r].astype(np.float32))
                idx = torch.rand(f.shape[:2]).argsort(-1)[:, :a.k].sort(-1).values       # a random subset of the windows, as in training
                f = f.gather(1, idx.unsqueeze(-1).expand(-1, -1, f.shape[-1]))
                loss = (F.binary_cross_entropy_with_logits(net(f), torch.from_numpy(tgt[r]), reduction="none") * torch.from_numpy(wgt[r])).mean()
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
                trace.append(f"ep{ep + 1} gold {per(gold_y, predict(ema, go)).mean():.4f} watch {per(val_y, predict(ema, va)).mean():.4f}")
        return ema, trace

    for policy in a.policies:
        t0 = time.time()
        ft = policy.startswith("ft:")
        assert not ft or own is not None, "ft: policies need --head"
        tgt, wgt = targets_for(policy[3:] if ft else policy, lab, tables, halves[a.fit_gold])
        preds = []
        for seed in a.seeds:
            ema, trace = fit(tgt, wgt, tr, seed, own if ft else None)
            p = predict(ema, ev)
            preds.append(p)
            print(f"  {policy:20s} seed {seed}: gold {per(gold_y, p[:len(go)]).mean():.4f}  watch fold vs consensus {per(val_y, p[len(go):]).mean():.4f}"
                  + ("  | " + ", ".join(trace) if trace else ""), flush=True)
            if a.save_heads:
                np.savez(OUT / f"head_{a.tag}_{policy.replace(':', '-')}_{a.fit_gold}_s{seed}.npz", **{k: v.numpy() for k, v in ema.state_dict().items()})
        mean = np.mean(preds, 0)
        pd.DataFrame(mean, index=lab.index[ev], columns=T).assign(set=sets).to_csv(OUT / f"pred_{a.tag}_{policy.replace(':', '-')}_{a.fit_gold}.csv")
        print(f"{policy:20s} fit-gold {a.fit_gold}: {len(a.seeds)} seeds together: gold {per(gold_y, mean[:len(go)]).mean():.4f}, "
              f"watch fold vs consensus {per(val_y, mean[len(go):]).mean():.4f}  ({time.time() - t0:.0f}s)", flush=True)


def compare(a):
    lab = ld("runs/solo/gpjw_labels_v0.csv")
    gid = lab.index[lab.is_gold == 1]
    y = lab.loc[gid, [f"gold_{c}" for c in T]].values.astype(int)
    halves = gold_halves(gid)
    g = lambda f: (lambda d: d[d.set == "gold"].loc[gid, T].values)(ld(f))
    P = {f.name[len(f"pred_{a.tag}_"):-4]: g(f) for f in sorted(OUT.glob(f"pred_{a.tag}_*.csv"))}
    ref = P.pop("checkpoint")
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(y), len(y)) for _ in range(3000)]
    rk = lambda x: np.apply_along_axis(lambda v: pd.Series(v).rank().values, 0, x) / len(x)
    rows = [{"head": "checkpoint's own", "gold-58": per(y, ref).mean()}]
    table = {"checkpoint's own": per(y, ref)}
    for n, p in P.items():
        for name, q in ((n, p), (n + " + own (mean)", (rk(p) + rk(ref)) / 2)):
            d = np.array([np.nanmean(per(y, q, b)) - np.nanmean(per(y, ref, b)) for b in boots])
            r = {"head": name, "gold-58": per(y, q).mean(), "vs own": per(y, q).mean() - per(y, ref).mean(),
                 "CI low": np.percentile(d, 2.5), "CI high": np.percentile(d, 97.5), "P>0": (d > 0).mean()}
            fit = n.rsplit("_", 1)[-1]
            if fit in ("A", "B"):                  # the mapping saw this half of gold: only the other half counts
                o = np.where(~halves[fit])[0]
                r["other half"], r["own, same half"] = np.nanmean(per(y, q, o)), np.nanmean(per(y, ref, o))
            rows.append(r)
            table[name] = per(y, q)
    print(f"gold-58, heads on the {a.tag} features (plain view); '+ own' = rank mean with the checkpoint's own head")
    print(pd.DataFrame(rows).set_index("head").round(4).to_string())
    print("\nper finding")
    print(pd.DataFrame(table, index=T).round(3).T.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--feats", required=True)
    r.add_argument("--tag", required=True, help="short name of the backbone, e.g. a384 or b224")
    r.add_argument("--policies", nargs="+", required=True)
    r.add_argument("--head", default="", help="npz of the checkpoint's own head weights, for ft: policies")
    r.add_argument("--fit-gold", default="all", choices=["all", "A", "B"])
    r.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    r.add_argument("--epochs", type=int, default=6)
    r.add_argument("--bs", type=int, default=16)
    r.add_argument("--k", type=int, default=24, help="windows per study in a training step (of the 94)")
    r.add_argument("--lr", type=float, default=1e-3)
    r.add_argument("--wd", type=float, default=0.05)
    r.add_argument("--ema", type=float, default=0.998)
    r.add_argument("--watch-fold", type=int, default=3)
    r.add_argument("--all-train", action="store_true", help="train on every report-labelled study (the watch fold is then not held out)")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--trace", action="store_true")
    r.add_argument("--save-heads", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--tag", required=True)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else compare(a)
