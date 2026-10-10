"""Train one fold of the public 2.5D reader recipe (goodpjw2008 `train.py`) on a Kaggle T4.

The model, window sampling and augmentation come from the reader's own `knee.py`, unchanged (the
series are read from our packed cache shards through its `loader` hook); the
loss, optimiser, schedule, gradient clipping and EMA follow the public `train.py`. What differs,
because of the hardware:
  - fp16 autocast with a GradScaler (a T4 has no native bf16);
  - the batch of --bs studies is run --micro studies at a time with gradient accumulation
    (ConvNeXt has no BatchNorm, so the gradient is the same up to dropout noise);
  - validation runs every --eval-every epochs and after the last one;
  - a timing run before training lowers --epochs when the session budget is too small for them.
The checkpoint has the public format ({"model": EMA weights, "args": ...}), so the reader's
`infer.py` loads it as it is.

Usage: python train_gpjw.py --tag ctrl --fold 3 --index JSON --labels CSV --meta CSV --series-csv CSV --out DIR
"""
import argparse, copy, json, math, os, time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from knee import LABELS, KneeNet, StudyDataset, build_study_table
from shards import ShardLoader


def macro_auc(y, p):
    aucs = [roc_auc_score(y[:, i], p[:, i]) if 0 < y[:, i].sum() < len(y) else np.nan for i in range(y.shape[1])]
    return float(np.nanmean(aucs)), [float(a) for a in aucs]


def seed_worker(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)   # numpy drives the window sampling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--index", required=True, help="json: series -> shard file (written by the session cell)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--series-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="convnext_tiny")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--k_eval", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_head", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.02)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--budget-h", type=float, default=0, help="hours this job may take; 0 = no limit")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--init", default="", help="checkpoint to start from (public format); fine-tuning")
    ap.add_argument("--warm", type=float, default=1.0, help="warm-up length in epochs")
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many training studies")
    args = ap.parse_args()
    assert args.bs % args.micro == 0, "--bs must be a multiple of --micro"
    t_start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = dev == "cuda"
    torch.backends.cudnn.benchmark = True      # a micro-batch holds 1..6*micro series: few distinct shapes
    os.makedirs(args.out, exist_ok=True)
    name = f"{args.tag}_fold{args.fold}"
    log = open(f"{args.out}/log_{name}.txt", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say(json.dumps(vars(args)))
    lab = pd.read_csv(args.labels, index_col=0)
    lab.index = lab.index.astype(str)
    meta = pd.read_csv(args.meta, dtype={"study": str, "series": str})
    series = pd.read_csv(args.series_csv, dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    table = build_study_table(series, dict(zip(meta.series, meta.n.fillna(0))))
    loader = ShardLoader(args.index)

    gold = lab[lab.is_gold == 1]
    weak = lab[lab.is_gold == 0]
    tr = weak[weak.fold != args.fold]
    va = weak[weak.fold == args.fold]
    if args.limit:
        tr = tr.iloc[:args.limit]
    say(f"train {len(tr)}  val {len(va)}  gold {len(gold)}  studies without a usable series: "
        f"{int((~lab.index.isin(list(table))).sum())}")

    gold_y = gold[[f"gold_{c}" for c in LABELS]].values
    # a mixed-label table keeps the original labels in orig_* columns: the held-out fold is scored on those
    val_cols = [f"orig_{c}" for c in LABELS] if f"orig_{LABELS[0]}" in lab.columns else LABELS
    val_y = (va[val_cols].values > 0.5).astype(int) if len(va) else None

    model = KneeNet(args.backbone, pretrained=not (args.no_pretrained or args.init)).to(dev).to(memory_format=torch.channels_last)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location="cpu", weights_only=False)["model"])
        say(f"initialised from {os.path.basename(args.init)}")
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    enc_params = list(model.enc.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.lr_head}],
                            weight_decay=args.wd)
    scaler = torch.amp.GradScaler(dev, enabled=amp)

    def fits(micro):
        """One training step on the largest batch a loader can produce: every study with all six series."""
        torch.cuda.reset_peak_memory_stats()
        try:
            x = torch.zeros((micro, 6, args.k, 3, 384, 384), dtype=torch.uint8, device=dev)
            with torch.autocast(dev, dtype=torch.float16):
                logit = model(x, torch.ones((micro, 6), dtype=torch.bool, device=dev),
                              torch.zeros((micro, 6, args.k), device=dev), res=args.res, train=True)
            scaler.scale(F.binary_cross_entropy_with_logits(logit.float(), torch.zeros_like(logit.float()))).backward()
            peak = torch.cuda.max_memory_allocated()
            ok = peak < 0.85 * torch.cuda.get_device_properties(0).total_memory
            say(f"memory check: micro {micro} needs {peak / 1e9:.1f} GB -> {'fits' if ok else 'too close to the limit'}")
        except torch.OutOfMemoryError:
            ok = False
            say(f"memory check: micro {micro} does not fit")
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return ok

    if amp:
        model.train()
        while args.micro > 1 and not fits(args.micro):
            args.micro //= 2
    acc = args.bs // args.micro
    dl_kw = dict(num_workers=args.workers, pin_memory=amp, worker_init_fn=seed_worker)
    # a table with w_<finding> columns gives every cell a loss weight; the weights travel next to the targets
    w_cols = [f"w_{c}" for c in LABELS]
    w_tr = tr[w_cols].values if w_cols[0] in tr.columns else np.ones((len(tr), len(LABELS)))
    train_ds = StudyDataset(tr.index, table, None, np.concatenate([tr[LABELS].values, w_tr], 1).astype(np.float32),
                            args.k, train=True, loader=loader)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=args.micro, shuffle=True, drop_last=True,
                                           persistent_workers=args.workers > 0, **dl_kw)
    eval_sets = {"gold": gold}
    if len(va):
        eval_sets["val"] = va
    eval_dl = {k: torch.utils.data.DataLoader(StudyDataset(v.index, table, None, None, args.k_eval, loader=loader),
                                              batch_size=2 * args.micro, shuffle=False, **dl_kw)
               for k, v in eval_sets.items()}
    steps_per_epoch = len(train_dl) // acc

    def forward_loss(x, mask, pos, y):
        x, mask, pos, y = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev), y.to(dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=amp):
            logit = model(x, mask, pos, res=args.res, train=True)
        y, w = y[:, :len(LABELS)], y[:, len(LABELS):]
        return (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).mean() / acc

    @torch.no_grad()
    def predict(net, loader):
        net.eval()
        out = []
        for x, mask, pos, _ in loader:
            x, mask, pos = x.to(dev, non_blocking=True), mask.to(dev), pos.to(dev)
            with torch.autocast(dev, dtype=torch.float16, enabled=amp):
                out.append(net(x, mask, pos, res=args.res).float().sigmoid().cpu())
        return torch.cat(out).numpy()

    # ---- timing run: gradients are computed and thrown away, the weights do not change ----
    epochs = args.epochs
    if args.budget_h:
        model.train()
        sub = torch.utils.data.Subset(train_ds, range(min(len(train_ds), args.micro * acc * 9)))
        bench_dl = torch.utils.data.DataLoader(sub, batch_size=args.micro, drop_last=True, num_workers=args.workers,
                                               worker_init_fn=seed_worker)
        times, t0 = [], time.time()
        for i, batch in enumerate(bench_dl):
            scaler.scale(forward_loss(*batch)).backward()
            if (i + 1) % acc == 0:
                opt.zero_grad(set_to_none=True)
                if amp:
                    torch.cuda.synchronize()
                times.append(time.time() - t0)
                t0 = time.time()
        opt.zero_grad(set_to_none=True)
        t_step = float(np.median(times[2:])) if len(times) > 3 else float(np.mean(times))
        n_eval = sum(len(v) for v in eval_sets.values())
        t0 = time.time()
        n_done = 0
        for j, (x, mask, pos, _) in enumerate(eval_dl["gold"]):          # also warms the eval shapes
            with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=amp):
                model(x.to(dev), mask.to(dev), pos.to(dev), res=args.res)
            n_done += len(x)
        if amp:
            torch.cuda.synchronize()
        t_eval = (time.time() - t0) / max(n_done, 1) * n_eval
        t_epoch = steps_per_epoch * t_step
        left = args.budget_h * 3600 - (time.time() - t_start) - 600      # ten minutes kept for saving
        fit = int((left - t_eval * (1 + args.epochs // args.eval_every)) // t_epoch)
        peak = torch.cuda.max_memory_allocated() / 1e9 if amp else 0.0
        say(f"timing: {t_step:.2f} s/step, {t_epoch / 60:.1f} min/epoch, {t_eval / 60:.1f} min per evaluation, "
            f"peak GPU memory {peak:.1f} GB; the budget fits {fit} epochs")
        if fit < epochs:
            epochs = max(1, fit)
            say(f"WARNING: epochs lowered from {args.epochs} to {epochs} to fit the {args.budget_h} h budget")
        model.train()

    steps = epochs * steps_per_epoch
    warm = max(1, int(steps_per_epoch * args.warm))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))

    step, preds, history = 0, None, []
    for ep in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        t0, tot, n, k = time.time(), 0.0, 0, 0
        for batch in train_dl:
            loss = forward_loss(*batch)
            scaler.scale(loss).backward()
            tot += loss.item()
            k += 1
            if k < acc:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            k = 0
            d = min(args.ema, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), model.parameters()):
                    pe.lerp_(pm, 1 - d)
                for be, bm in zip(ema.buffers(), model.buffers()):
                    be.copy_(bm)
            n += 1
            if n % 100 == 0:
                print(f"  ep {ep} step {n}/{steps_per_epoch} loss {tot / n:.4f} {(time.time() - t0) / n:.2f}s/it", flush=True)
        msg = f"ep {ep:2d} loss {tot / max(n, 1):.4f} time {time.time() - t0:.0f}s"
        row = {"epoch": ep, "loss": tot / max(n, 1), "seconds": time.time() - t0}
        if (ep + 1) % args.eval_every == 0 or ep == epochs - 1:
            preds = {key: predict(ema, dl) for key, dl in eval_dl.items()}
            g_auc, g_each = macro_auc(gold_y, preds["gold"])
            msg += f" | gold {g_auc:.4f}"
            row["gold"] = g_auc
            if val_y is not None:
                v_auc, v_each = macro_auc(val_y, preds["val"])
                msg += f" | val {v_auc:.4f} [" + " ".join(f"{a:.3f}" for a in v_each) + "]"
                row["val"], row["val_each"] = v_auc, v_each
        history.append(row)
        say(msg)

    ck_args = {**vars(args), "epochs": epochs, "epochs_asked": args.epochs}
    torch.save({"model": ema.state_dict(), "args": ck_args}, f"{args.out}/{name}.pt")
    rows = []
    for key, v in eval_sets.items():
        df = pd.DataFrame(preds[key], index=v.index, columns=LABELS)   # probabilities
        df.insert(0, "set", key)
        rows.append(df)
    pd.concat(rows).to_csv(f"{args.out}/oof_{name}.csv")
    g_auc, g_each = macro_auc(gold_y, preds["gold"])
    result = {"tag": args.tag, "fold": args.fold, "args": ck_args, "gold": g_auc, "gold_each": dict(zip(LABELS, g_each)),
              "history": history, "minutes": (time.time() - t_start) / 60}
    if val_y is not None:
        result["val"], v_each = macro_auc(val_y, preds["val"])
        result["val_each"] = dict(zip(LABELS, v_each))
    json.dump(result, open(f"{args.out}/results_{name}.json", "w"), indent=1)
    say("gold per class: " + " ".join(f"{c[:6]}={a:.3f}" for c, a in zip(LABELS, g_each)))
    say(f"done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    main()
