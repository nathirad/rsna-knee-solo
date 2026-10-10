# %%
# ---- solo driver (new): test slot cache -> own members -> submission.csv ----
# Runs after the preprocessing cells and solo_model.py in one script (see build.py).
# No public ensemble behind it: every test study gets a row here or the run fails loudly.
import fnmatch as _fnmatch
import threading as _threading

OUT_CSV = Path(os.environ.get("SOLO_OUT", "/kaggle/working/submission.csv"))
RECEIPT = OUT_CSV.with_name("_solo_receipt.json")
ASSET_ROOT = os.environ.get("SOLO_ASSETS", "/kaggle/input")
_T = {"start": time.time()}


# What the E3 members (the plain-list members.json of rsna-knee-own-members) were fitted
# on: rsna-knee-cache-full, i.e. 336 px, 3 slices, band 0.20-0.80.
E3_CACHE = {"img": 336, "slices": 3, "crop_mm": 130.0, "band": [0.20, 0.80]}


def _find_json(root):
    """member_*.json and members.json under root, without walking the DICOM trees (the
    test tree alone is some 200,000 files, and this runs inside the scored time)."""
    hits = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        hits += [os.path.join(d, f) for f in files
                 if f == "members.json" or _fnmatch.fnmatch(f, "member_*.json")]
    return sorted(hits)


def _load_members(root):
    """Members from `member_*.json` (one per trained fold) and/or a `members.json`."""
    found = []
    for p in _find_json(root):
        doc = json.loads(Path(p).read_text())
        if Path(p).name != "members.json":
            found.append({**doc, "path": str(Path(p).with_name(doc["file"]))})
            continue
        cache = doc.get("cache") if isinstance(doc, dict) else E3_CACHE
        for m in (doc["members"] if isinstance(doc, dict) else doc):
            # An E3-era entry has no window mode: it was trained one window at a time.
            m = {"mode": "one", "groups": 1, **m}
            if cache is not None:
                m.setdefault("cache", cache)
            found.append({**m, "path": str(Path(p).with_name(m["file"]))})
    keep = os.environ.get("SOLO_ARCHS")        # optional filter, e.g. "cnx"
    if keep:
        found = [m for m in found if m["arch"] in keep.split(",")]
    # optional field filter, e.g. SOLO_KEEP="mode=one,labels=pilk": a training kernel's
    # output can hold members of more than one experiment
    for pair in filter(None, os.environ.get("SOLO_KEEP", "").split(",")):
        k, v = pair.split("=", 1)
        found = [m for m in found if str(m.get(k)) == v]
    return found


MEMBERS = _load_members(ASSET_ROOT)
if not MEMBERS:
    raise FileNotFoundError(f"no member_*.json or members.json under {ASSET_ROOT}")

# The cache contract: the pixels a member is given here must be the pixels it was fitted
# on. A mismatch changes no shape that would raise by itself, so it is checked by name.
HERE = {"img": int(CACHE_IMG), "slices": int(CACHE_SLICES), "crop_mm": float(CROP_MM),
        "band": [float(b) for b in SLICE_BAND]}
for _m in MEMBERS:
    _c = _m.get("cache")
    if _c is None:
        raise RuntimeError(f"{_m['file']}: no cache contract recorded for this member")
    _c = {"img": int(_c["img"]), "slices": int(_c["slices"]), "crop_mm": float(_c["crop_mm"]),
          "band": [float(b) for b in _c["band"]]}
    if _c != HERE:
        raise RuntimeError(f"{_m['file']} was fitted on cache {_c} but this script decodes {HERE}; "
                           f"rebuild the notebook with build.py for that contract")
log(f"solo: {len(MEMBERS)} member(s); cache contract {HERE}")

# Scoring always reads the test split. SOLO_SPLIT=train with a study list is for check runs:
# it sends training studies through this exact path, to compare with the training cache.
SPLIT = os.environ.get("SOLO_SPLIT", "test")
if SPLIT == "test":
    test_df = pd.read_csv(ROOT / "test.csv")
    hte = annotate(walk("test_series"))
else:
    test_df = pd.read_csv(os.environ["SOLO_STUDY_CSV"], dtype={"StudyInstanceUID": str})[["StudyInstanceUID"]]
    _items = [(f"{SPLIT}_series", st, e.name, e.path) for st in test_df["StudyInstanceUID"]
              for e in os.scandir(ROOT / f"{SPLIT}_series" / st) if e.is_dir()]
    with ThreadPoolExecutor(max_workers=HDR_THREADS) as _pool:
        hte = annotate(pd.DataFrame(list(_pool.map(probe, _items))))
test_series = pd.read_csv(ROOT / f"{SPLIT}_series.csv")
plane_map = dict(zip(test_series["SeriesInstanceUID"], test_series["Anatomical_Plane"]))
_T["header"] = time.time()
log(f"header pass: {len(hte)} {SPLIT} series")
slots_te = pick_slots(hte, plane_map)
st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, "test "), "test")
_T["cache"] = time.time()

N_GPU = torch.cuda.device_count()
DEVICES = [torch.device(f"cuda:{i}") for i in range(N_GPU)] or [torch.device("cpu")]
EVAL_BATCH = int(os.environ.get("SOLO_BATCH", "16"))


def _run_member(mem, dev):
    model = SoloModel(mem.get("timm_name") or solo_timm_name(mem["arch"]), N_SLOT,
                      int(mem["groups"]), len(TARGETS), pretrained=False)
    sd = torch.load(mem["path"], map_location="cpu")
    missing, unexpected = model.load_state_dict({k: v.float() if v.is_floating_point() else v
                                                 for k, v in sd.items()}, strict=False)
    # E3 weights predate the window embedding; it stays at zero, which is what they saw.
    if unexpected or set(missing) - {"head.win_emb"}:
        raise RuntimeError(f"{mem['file']}: state dict mismatch, missing {missing}, unexpected {unexpected}")
    model = model.to(dev).eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(st_te), EVAL_BATCH):
            x = torch.from_numpy(np.ascontiguousarray(Cte[b:b + EVAL_BATCH])).to(dev)
            m = torch.from_numpy(Mte[b:b + EVAL_BATCH]).to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                z = solo_logits(model, x, m, int(mem["img"]), mem["mode"]).float()
            out.append(torch.sigmoid(z).cpu().numpy())
    p = np.concatenate(out) if out else np.zeros((0, len(TARGETS)), np.float32)
    if not np.isfinite(p).all():
        raise RuntimeError(f"{mem['file']}: non-finite predictions")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return p


# One worker per GPU, each taking members off a shared list: with two T4s the folds run
# two at a time. A failure in any worker is re-raised after the others finish.
PROBS, _ERRORS = [None] * len(MEMBERS), []
_todo, _lock = list(range(len(MEMBERS))), _threading.Lock()


def _worker(dev):
    while True:
        with _lock:
            if not _todo or _ERRORS:
                return
            k = _todo.pop(0)
        try:
            PROBS[k] = _run_member(MEMBERS[k], dev)
            log(f"member {MEMBERS[k]['file']} done on {dev}")
        except Exception as exc:              # noqa: BLE001 - reported below
            _ERRORS.append((MEMBERS[k]["file"], exc))


_threads = [_threading.Thread(target=_worker, args=(d,)) for d in DEVICES]
for _th in _threads:
    _th.start()
for _th in _threads:
    _th.join()
if _ERRORS:
    raise RuntimeError(f"member {_ERRORS[0][0]} failed") from _ERRORS[0][1]
_T["infer"] = time.time()
if os.environ.get("SOLO_DUMP"):      # check runs only (smoke_local.py, solo-check); unset when scoring
    np.savez(os.environ["SOLO_DUMP"], studies=np.array(st_te), cache=Cte, mask=Mte,
             probs=np.stack(PROBS), members=np.array([m["file"] for m in MEMBERS]))

# Members are combined as the mean of per-finding percentile ranks. The metric is AUC, so
# only the order within a column matters, and ranks keep one member's scale from dominating.
ranks = [pd.DataFrame(p, columns=TARGETS).rank(pct=True).values for p in PROBS]
sub = pd.DataFrame(np.mean(ranks, axis=0), columns=TARGETS)
sub.insert(0, "StudyInstanceUID", st_te)
sub = test_df[["StudyInstanceUID"]].astype(str).merge(sub.astype({"StudyInstanceUID": str}),
                                                       on="StudyInstanceUID", how="left")
n_missing = int(sub[TARGETS].isna().any(axis=1).sum())
# A study with no usable series has no prediction. It gets the middle rank: no information,
# and a valid file. More than a handful means the reader is broken, not the studies.
sub[TARGETS] = sub[TARGETS].fillna(0.5)
if len(test_df) > 20 and n_missing > 0.05 * len(test_df):
    raise RuntimeError(f"{n_missing}/{len(test_df)} test studies have no prediction")
assert sub.columns.tolist() == ["StudyInstanceUID", *TARGETS], "submission schema"
assert len(sub) == len(test_df) and sub["StudyInstanceUID"].is_unique, "submission rows"
assert np.isfinite(sub[TARGETS].values).all(), "submission has non-finite values"
sub.to_csv(OUT_CSV, index=False)
_T["end"] = time.time()
RECEIPT.write_text(json.dumps({
    "n_studies": len(test_df), "n_without_prediction": n_missing, "members": [m["file"] for m in MEMBERS],
    "slots_filled": float(Mte.mean()) if len(Mte) else None, "decode_failed": len(DECODE_FAILED),
    "gpus": N_GPU, "cache": HERE,
    "seconds": {"header": _T["header"] - _T["start"], "cache": _T["cache"] - _T["header"],
                "inference": _T["infer"] - _T["cache"], "total_script": _T["end"] - T0}}, indent=1))
log(f"solo wrote {OUT_CSV}: {len(sub)} rows, {n_missing} without a prediction, "
    f"{_T['end'] - T0:.0f}s in total")
