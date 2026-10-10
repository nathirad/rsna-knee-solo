"""Solo member training: one strong single model per fold, for a standalone submission.

Grown from colab/e2_label_ab.py (the E2/E3 job; pilkwang/rsna-knee-baseline-v1 recipe,
Apache-2.0). What is new here:

  * every slice window of a slot is a token of one forward pass (WINDOWS=all), instead of
    one random three-slice window per step;
  * slot dropout, EMA weights, gradient accumulation, absent slots are not encoded;
  * a second label arm (LABELS=pseudo) that mixes out-of-fold predictions of an earlier
    round into the report labels;
  * each fold writes its holdout predictions (the out-of-fold table), a fp16 checkpoint and
    a `member_*.json` that carries the cache contract the weights were fitted on.

Validation is unchanged (see PLAN.md): the 58 gold studies are never trained on; folds are
grouped by report text; the holdout fold is scored against the binarised stevenleehans v4
table on its confident cells; the epoch count is fixed, nothing is selected on gold-58.

Everything is configured through the environment (defaults are the recommended values):
  FOLD=0 SEED=0 ARCH=cnx IMG=224 EPOCHS=25 WINDOWS=all LABELS=pilk ALPHA=0.5
  BATCH_STUDIES=4 ACCUM=2 SLOT_DROP=0.15 EMA=0.999 EVAL_EVERY=5 PRETRAINED=1
  IN_DIR RES_DIR CACHE_DIR PSEUDO_CSV BENCH_ONLY
"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# solo_model.py sits next to this file on Kaggle (the notebook writes both) and in
# ../rsna-knee-solo/ in the repo.
_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE.parent / "rsna-knee-solo")]
from solo_model import SoloModel, solo_logits, solo_timm_name   # noqa: E402

T0 = time.time()
FOLD = int(os.environ.get("FOLD", "0"))
SEED = int(os.environ.get("SEED", "0"))
ARCH = os.environ.get("ARCH", "cnx")
IMG = int(os.environ.get("IMG", "224"))
EPOCHS = int(os.environ.get("EPOCHS", "25"))
WINDOWS = os.environ.get("WINDOWS", "all")            # all | one
LABELS = os.environ.get("LABELS", "pilk")             # pilk | pseudo | pseudo_keep
ALPHA = float(os.environ.get("ALPHA", "0.5"))         # share of the out-of-fold prediction
BATCH_STUDIES = int(os.environ.get("BATCH_STUDIES", "4"))
ACCUM = int(os.environ.get("ACCUM", "2"))             # optimizer step every ACCUM batches
SLOT_DROP = float(os.environ.get("SLOT_DROP", "0.15"))
EMA_DECAY = float(os.environ.get("EMA", "0.999"))     # 0 switches EMA off
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "5"))
PRETRAINED = os.environ.get("PRETRAINED", "1") == "1"
BENCH_ONLY = os.environ.get("BENCH_ONLY") == "1"

IN = Path(os.environ.get("IN_DIR", "/kaggle/working/in"))
RES = Path(os.environ.get("RES_DIR", "/kaggle/working/res")); RES.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(os.environ.get("CACHE_DIR", "/kaggle/working/cache"))
PSEUDO_CSV = os.environ.get("PSEUDO_CSV", "")

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
N_SLOT, GROUP, N_FOLD = 6, 3, 5
EVAL_BATCH = 16
LR_HEAD, LR_BACKBONE, WEIGHT_DECAY = 1e-3, 1e-4, 0.02
AUG_ROT_DEG, AUG_SCALE, AUG_SHIFT, AUG_INTENSITY = 8.0, 0.08, 0.05, 0.10
assert WINDOWS in ("all", "one") and LABELS in ("pilk", "pseudo", "pseudo_keep")
TAG = f"{ARCH.replace(':', '-')}_{IMG}_{LABELS}_{WINDOWS}_f{FOLD}_s{SEED}"


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] f{FOLD} {msg}", flush=True)


# ---------------------------------------------------------------- cache
class ShardCache:
    """The slot cache as it is stored: one .npy, or shards of 500 studies.

    Shards stay memory-mapped and are indexed in place. Joining them first would need the
    whole array in RAM once per process, and two fold processes share one machine.
    """
    def __init__(self, paths):
        self.parts = [np.load(p, mmap_mode="r") for p in paths]
        self.offsets = np.cumsum([0] + [len(p) for p in self.parts])
        self.shape = (int(self.offsets[-1]),) + self.parts[0].shape[1:]

    def take(self, idx, lo=0, hi=None):
        """Studies `idx`, slices lo:hi of every slot (all slices by default)."""
        hi = self.shape[2] if hi is None else hi
        out = np.empty((len(idx), self.shape[1], hi - lo) + self.shape[3:], np.uint8)
        part = np.searchsorted(self.offsets, idx, side="right") - 1
        for k, (i, p) in enumerate(zip(idx, part)):
            out[k] = self.parts[p][i - self.offsets[p], :, lo:hi]
        return out


meta = json.loads((CACHE_DIR / "cache_meta.json").read_text())
_parts = sorted(CACHE_DIR.glob("cache_*_part*.npy"))
_fulls = [p for p in sorted(CACHE_DIR.glob("cache_*.npy")) if "_part" not in p.name]
C = ShardCache(_fulls[:1] or _parts)
M = np.load(CACHE_DIR / "mask.npy")
studies = meta["studies"]
assert C.shape[0] == len(studies) == len(M), "cache, mask and study list disagree"
N_GROUP = C.shape[2] // GROUP
assert C.shape[2] == meta["slices"] == N_GROUP * GROUP, "cache slices do not match cache_meta.json"
G_MODEL = N_GROUP if WINDOWS == "all" else 1
CONTRACT = {"img": int(meta["img"]), "slices": int(meta["slices"]),
            "crop_mm": float(meta["crop_mm"]), "band": [float(b) for b in meta["band"]]}
log(f"cache {C.shape}, slots filled {M.mean():.3f}, {N_GROUP} window(s) of {GROUP}; contract {CONTRACT}")

dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
ON_GPU = dev.type == "cuda"
if ON_GPU:
    gpu = torch.cuda.get_device_name(0)
    # T4 reports bf16 as supported but emulates it (slow, and it cost 0.14 AUC for one team)
    AMP = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    # Batch sizes are bucketed in SoloModel, so autotune runs for a handful of sizes only.
    torch.backends.cudnn.benchmark = True
else:
    gpu, AMP = "cpu", None
log(f"device {gpu}, amp {AMP}")


def autocast():
    return torch.autocast("cuda", dtype=AMP, enabled=ON_GPU)


# ---------------------------------------------------------------- labels and folds
train_df = pd.read_csv(IN / "train.csv")
gold = train_df.set_index("StudyInstanceUID")[TARGETS]
gold = gold[gold.notna().all(axis=1)]
rep = train_df.set_index("StudyInstanceUID")["Report"].fillna("")


def report_table():
    t = pd.read_csv(IN / "report_labels_v2.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
    conf = t[[c + "__conf" for c in TARGETS]].values
    return t[TARGETS], pd.DataFrame(conf, index=t.index, columns=TARGETS)


def ref_table():
    t = pd.read_csv(IN / "llm_labels_v4_blend.csv").drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")[TARGETS]
    return t, (2 * t - 1).abs()


def fold_of(study):
    """Folds grouped by report text: studies with the same report share a fold."""
    return int(hashlib.md5(rep.get(study, study).encode()).hexdigest()[:8], 16) % N_FOLD


ref_p, ref_c = ref_table()
idx_gold = np.array([i for i, s in enumerate(studies) if s in gold.index])
non_gold = [i for i, s in enumerate(studies) if s not in gold.index]
fold = np.array([fold_of(studies[i]) for i in non_gold])
idx_va = np.array([i for i, f in zip(non_gold, fold) if f == FOLD])
idx_tr_all = np.array([i for i, f in zip(non_gold, fold) if f != FOLD])
gold_y = gold.loc[[studies[i] for i in idx_gold]].values.astype(int)
va_ref = ref_p.reindex([studies[i] for i in idx_va])
va_y = (va_ref.values > 0.5).astype(int)
va_ok = (ref_c.reindex([studies[i] for i in idx_va]).values >= 0.5)
log(f"gold {len(idx_gold)} | holdout {len(idx_va)} | train {len(idx_tr_all)}")


def targets():
    """Soft targets Y and cell weights W for the training studies of this fold."""
    p_tab, c_tab = report_table()
    tr = np.array([i for i in idx_tr_all if studies[i] in p_tab.index])
    names = [studies[i] for i in tr]
    Y = np.zeros((len(studies), len(TARGETS)), np.float32)
    W = np.zeros_like(Y)
    Y[tr] = p_tab.loc[names].values
    W[tr] = 0.25 + 0.75 * c_tab.loc[names].values
    if LABELS.startswith("pseudo"):
        # Out-of-fold predictions of an earlier round, mixed into the report label. The
        # report is silent on many findings (Synovitis in 4 of 5 reports); the image model
        # saw the image, so its out-of-fold opinion fills the silence. Kept fractional:
        # rounding to 0/1 was reported to hurt.
        oof = pd.read_csv(PSEUDO_CSV).drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")
        has = np.array([s in oof.index for s in names])
        if has.mean() < 0.95:
            raise RuntimeError(f"pseudo table covers only {has.mean():.1%} of the training studies")
        cov = [s for s, h in zip(names, has) if h]
        # An out-of-fold table from a different split is not out of fold for this one.
        wrong = [s for s in cov if int(oof.at[s, "fold"]) != fold_of(s)]
        if wrong:
            raise RuntimeError(f"pseudo table fold differs from this split for {len(wrong)} studies")
        P = oof.loc[cov, TARGETS].values.astype(np.float32)
        if not (np.isfinite(P).all() and P.min() >= 0 and P.max() <= 1):
            raise RuntimeError("pseudo table must hold probabilities in [0, 1]")
        sel = tr[has]
        Y[sel] = ALPHA * P + (1 - ALPHA) * Y[sel]
        # `pseudo` also raised the cell weight toward 1. Where the report is silent (weight
        # ~0.29) that more than doubled the pull toward the member's own earlier guess, and
        # gold-58 fell most on the findings reports mention least (Synovitis, Baker's,
        # Fracture). `pseudo_keep` mixes the target only and leaves the weights alone.
        if LABELS == "pseudo":
            W[sel] = ALPHA + (1 - ALPHA) * W[sel]
        log(f"pseudo labels: alpha {ALPHA}, {has.sum()}/{len(tr)} training studies mixed")
    return Y, W, tr


def macro_auc(y, p, ok=None):
    from sklearn.metrics import roc_auc_score
    out = []
    for j in range(y.shape[1]):
        m = np.ones(len(y), bool) if ok is None else ok[:, j]
        out.append(roc_auc_score(y[m, j], p[m, j]) if len(set(y[m, j])) > 1 else np.nan)
    return float(np.nanmean(out)), [None if np.isnan(v) else float(v) for v in out]


# ---------------------------------------------------------------- training pieces
def augment(imgs):
    """Rigid jitter and intensity scale, one draw per three-slice image. No flips: the
    knee has been mapped to a left-knee convention and five findings are side-defined."""
    lead = imgs.shape[:-3]
    x = imgs.reshape(-1, *imgs.shape[-3:]).float()
    n, d = x.shape[0], x.device
    rot = (torch.rand(n, device=d) - 0.5) * 2 * (AUG_ROT_DEG * np.pi / 180)
    sc = 1.0 + torch.rand(n, device=d) * AUG_SCALE
    tx = (torch.rand(n, device=d) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, device=d) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.zeros(n, 2, 3, device=d)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    x = (x * (1.0 + (torch.rand(n, 1, 1, 1, device=d) - 0.5) * 2 * AUG_INTENSITY)).clamp(0, 255)
    return x.reshape(*lead, *x.shape[-3:]).to(imgs.dtype)


def drop_slots(mask, p):
    """Hide each present slot with probability p, never all of a study's slots. Test
    studies come from sites with other protocols, so a member should not lean on one
    sequence being there."""
    if p <= 0:
        return mask
    keep = (torch.rand_like(mask) > p).float() * mask
    empty = keep.sum(1) == 0
    keep[empty] = mask[empty]
    return keep


class Ema:
    """Exponential moving average of the weights; the averaged model is what is saved."""
    def __init__(self, model, decay):
        self.decay, self.n = decay, 0
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)

    def state_dict(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        sd.update({k: v.clone() for k, v in self.shadow.items()})
        return sd


def build_model():
    return SoloModel(solo_timm_name(ARCH), N_SLOT, G_MODEL, len(TARGETS), pretrained=PRETRAINED)


@torch.no_grad()
def predict(model, idx):
    model.eval()
    out = []
    for b in range(0, len(idx), EVAL_BATCH):
        sel = idx[b:b + EVAL_BATCH]
        x = torch.from_numpy(C.take(sel)).to(dev)
        m = torch.from_numpy(M[sel]).to(dev)
        with autocast():
            z = solo_logits(model, x, m, IMG, WINDOWS).float()
        out.append(torch.sigmoid(z).cpu().numpy())
    out = np.concatenate(out)
    bad = ~np.isfinite(out)
    if bad.any():
        log(f"WARNING: {int(bad.sum())} non-finite predictions set to 0.5")
        out[bad] = 0.5
    return out


def evaluate(model):
    pg, pv = predict(model, idx_gold), predict(model, idx_va)
    g_auc, g_per = macro_auc(gold_y, pg)
    v_auc, v_per = macro_auc(va_y, pv, va_ok)
    return {"gold58": g_auc, "gold58_per_label": dict(zip(TARGETS, g_per)),
            "holdout": v_auc, "holdout_per_label": dict(zip(TARGETS, v_per))}, pg, pv


# ---------------------------------------------------------------- one fold
def run():
    Y, W, tr = targets()
    torch.manual_seed(SEED * 100 + FOLD); np.random.seed(SEED * 100 + FOLD)
    model = build_model().to(dev)
    opt = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
        {"params": model.head.parameters(), "lr": LR_HEAD}], weight_decay=WEIGHT_DECAY)
    steps_ep = len(tr) // BATCH_STUDIES // ACCUM
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[LR_BACKBONE, LR_HEAD],
                                                total_steps=max(EPOCHS * steps_ep, 1), pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=ON_GPU and AMP == torch.float16)
    ema = Ema(model, EMA_DECAY) if EMA_DECAY > 0 else None
    log(f"{TAG}: {len(tr)} train studies, {steps_ep} optimizer steps/epoch "
        f"(batch {BATCH_STUDIES} x accum {ACCUM}), windows {WINDOWS} ({G_MODEL} token(s)/slot)")
    hist, n_bad, t_run = [], 0, time.time()
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(tr)
        tot, nb, t_ep, t_io = 0.0, 0, time.time(), 0.0
        opt.zero_grad(set_to_none=True)
        for s in range(steps_ep * ACCUM):
            sel = np.sort(perm[s * BATCH_STUDIES:(s + 1) * BATCH_STUDIES])
            t_r = time.time()
            if WINDOWS == "one":      # one random window per step, as in E3
                g = np.random.randint(N_GROUP)
                x = torch.from_numpy(C.take(sel, g * GROUP, (g + 1) * GROUP)).to(dev)
            else:
                x = torch.from_numpy(C.take(sel)).to(dev)
            t_io += time.time() - t_r
            if BENCH_ONLY and s + 1 == 5 * ACCUM:     # the first steps carry start-up cost
                if ON_GPU:
                    torch.cuda.synchronize()
                t_b0, io_b0 = time.time(), t_io
            x = augment(x)
            m = drop_slots(torch.from_numpy(M[sel]).to(dev), SLOT_DROP)
            y = torch.from_numpy(Y[sel]).to(dev)
            w = torch.from_numpy(W[sel]).to(dev)
            with autocast():
                logit = solo_logits(model, x, m, IMG, "all")
            loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).mean()
            if not torch.isfinite(loss):
                n_bad += 1
                if n_bad <= 5 or n_bad % 50 == 0:
                    log(f"non-finite loss at epoch {ep + 1} batch {s} (total {n_bad}); batch skipped")
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss / ACCUM).backward()
            tot += loss.item(); nb += 1
            if (s + 1) % ACCUM == 0:
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(model)
            if BENCH_ONLY and s + 1 == 35 * ACCUM:
                if ON_GPU:
                    torch.cuda.synchronize()
                sps = (time.time() - t_b0) / 30       # optimizer steps 6..35
                r = {"gpu": gpu, "tag": TAG, "sec_per_optimizer_step": sps,
                     "read_sec_per_optimizer_step": (t_io - io_b0) / 30,
                     "first_5_steps_sec": t_b0 - t_ep,
                     "sec_per_epoch": sps * steps_ep, "hours_for_run": sps * steps_ep * EPOCHS / 3600,
                     "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if ON_GPU else None}
                log(f"BENCH {r}")
                (RES / f"bench_{TAG}.json").write_text(json.dumps(r))
                return
        row = {"epoch": ep + 1, "loss": tot / max(nb, 1), "sec": time.time() - t_ep, "read_sec": t_io}
        if (ep + 1) % EVAL_EVERY == 0 or ep + 1 == EPOCHS:
            ev, _, _ = evaluate(model)
            row.update(gold58=ev["gold58"], holdout=ev["holdout"])
            log(f"ep {ep + 1}/{EPOCHS} loss {row['loss']:.4f} gold58 {ev['gold58']:.4f} "
                f"holdout {ev['holdout']:.4f} {row['sec']:.0f}s")
        else:
            log(f"ep {ep + 1}/{EPOCHS} loss {row['loss']:.4f} {row['sec']:.0f}s")
        hist.append(row)

    raw, pg, pv = evaluate(model)
    final, which = raw, "last"
    if ema is not None:
        avg = build_model().to(dev)
        avg.load_state_dict(ema.state_dict(model))
        ev, pg_e, pv_e = evaluate(avg)
        log(f"last epoch: gold58 {raw['gold58']:.4f} holdout {raw['holdout']:.4f} | "
            f"EMA: gold58 {ev['gold58']:.4f} holdout {ev['holdout']:.4f}")
        # The EMA model is the one kept. The only exception is an average that is plainly
        # broken, which a 0.01 holdout drop on ~870 studies is and ordinary noise is not.
        if ev["holdout"] >= raw["holdout"] - 0.01:
            model, final, which, pg, pv = avg, ev, "ema", pg_e, pv_e
        else:
            log("WARNING: EMA is more than 0.01 below the last epoch on the holdout; keeping the last epoch")

    oof = pd.DataFrame(pv, columns=TARGETS, index=pd.Index([studies[i] for i in idx_va], name="StudyInstanceUID"))
    oof["fold"] = FOLD
    oof.to_csv(RES / f"oof_{TAG}.csv")
    pd.DataFrame(pg, columns=TARGETS, index=pd.Index([studies[i] for i in idx_gold], name="StudyInstanceUID")
                 ).to_csv(RES / f"gold_{TAG}.csv")
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()},
               RES / f"model_{TAG}.pt")
    member = {"file": f"model_{TAG}.pt", "arch": ARCH, "timm_name": solo_timm_name(ARCH), "img": IMG,
              "mode": WINDOWS, "groups": G_MODEL, "fold": FOLD, "seed": SEED, "labels": LABELS,
              "alpha": ALPHA if LABELS.startswith("pseudo") else None, "epochs": EPOCHS, "weights": which,
              "holdout": final["holdout"], "gold58": final["gold58"], "cache": CONTRACT}
    (RES / f"member_{TAG}.json").write_text(json.dumps(member, indent=1))
    (RES / f"results_{TAG}.json").write_text(json.dumps({
        "member": member, "gpu": gpu, "n_train": int(len(tr)), "n_holdout": int(len(idx_va)),
        "config": {"batch_studies": BATCH_STUDIES, "accum": ACCUM, "slot_drop": SLOT_DROP,
                   "ema": EMA_DECAY, "lr_backbone": LR_BACKBONE, "lr_head": LR_HEAD,
                   "weight_decay": WEIGHT_DECAY, "pretrained": PRETRAINED},
        "last_epoch": raw, "kept": final, "history": hist, "non_finite_batches": n_bad,
        "sec": time.time() - t_run}, indent=1))
    log(f"DONE {TAG}: kept {which}, holdout {final['holdout']:.4f}, gold58 {final['gold58']:.4f}, "
        f"{(time.time() - t_run) / 60:.1f} min")


run()
