

# ============================================================================
# Appended (ours): backbone features of chosen studies, for head-only retraining on a CPU.
# Everything above is the public notebook's cell, unchanged; only its final main() call is dropped.
# The model is backbone -> head (RaptorClassifier.encode / .head). This runs the same preprocessing
# and the same 94 windows as the scored pipeline (plain view, no mirror) and saves encode()'s output,
# together with the checkpoint's own logits from it (the check that the features are the right ones).
# ============================================================================
def extract_features():
    import pandas as pd
    t0 = time.time()
    split = os.environ.get("RAPTOR_SPLIT", "train")
    budget_s = float(os.environ.get("RAPTOR_BUDGET_H", "0")) * 3600
    out = os.environ.get("RAPTOR_OUT", "/kaggle/working/res")
    os.makedirs(out, exist_ok=True)
    ROOT = find_test_root()
    sdir = f"{ROOT}/{split}_series"
    ids = pd.read_csv(os.environ["RAPTOR_STUDIES"], dtype=str)["StudyInstanceUID"].tolist()
    ser = pd.read_csv(f"{ROOT}/{split}_series.csv")
    ser["StudyInstanceUID"] = ser["StudyInstanceUID"].astype(str)
    ser["SeriesInstanceUID"] = ser["SeriesInstanceUID"].astype(str)
    SER = {k: v.to_dict("records") for k, v in ser.groupby("StudyInstanceUID")}
    arm = ARMS[0]
    devs = [f"cuda:{g}" for g in range(torch.cuda.device_count())] or ["cpu"]
    models = {d: load_model(find_weight_file(arm["file"]), arm["arch"], arm["res"], d) for d in devs}
    N, pos = len(ids), {sid: i for i, sid in enumerate(ids)}
    feats, logits, filled = [None] * N, np.zeros((N, len(LAB)), np.float32), np.zeros(N, np.int16)
    print(f"{split}: {N} studies | devices {devs} | {time.time() - t0:.0f}s", flush=True)
    q = queue.Queue(maxsize=QUEUE_MAX); done_n = [0]; lock = threading.Lock()
    # At most QUEUE_MAX studies are prepared but not yet through the GPU. The public main() lets the pool run
    # ahead of the GPUs, which is fine for 1,322 test studies; with 4,407 the finished volumes (14 MB each)
    # piled up and the first full run was killed for memory after 1,000 studies.
    in_flight, stop = threading.Semaphore(QUEUE_MAX), threading.Event()

    def feed():
        for sid in ids:
            in_flight.acquire()
            if stop.is_set():
                return
            yield sid

    def consumer(d):
        model, res = models[d]
        while True:
            item = q.get()
            if item is None:
                q.task_done(); return
            sid, vol, mask = item; i = pos[sid]
            try:
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=d != "cpu"):
                    f = model.encode(gpu_windows(vol, mask, K_EVAL, res, d).unsqueeze(0))
                    z = model.head(f).float()
                feats[i], logits[i], filled[i] = f[0].to(torch.float16).cpu().numpy(), z[0].cpu().numpy(), int(mask.sum())
            except Exception as e:
                print(f"  [gpu {d}] study {i} {sid[:16]} FAILED ({type(e).__name__}: {e})", flush=True)
            with lock:
                done_n[0] += 1
                if done_n[0] % 200 == 0:
                    print(f"  [gpu] {done_n[0]}/{N} | {time.time() - t0:.0f}s", flush=True)
            in_flight.release()
            q.task_done()

    threads = [threading.Thread(target=consumer, args=(d,), daemon=True) for d in devs]
    for th in threads:
        th.start()
    import multiprocessing as mp
    global _SER, _TSDIR
    _SER, _TSDIR = SER, sdir
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
    with mp.get_context("fork").Pool(nproc, initializer=_pre_init) as pool:
        for n, (sid, vol, mask) in enumerate(pool.imap_unordered(_pre_one, feed(), chunksize=1)):
            q.put((sid, vol, mask))
            if budget_s and time.time() - t0 > budget_s:
                print(f"budget reached after {n + 1} studies: the rest is not saved", flush=True)
                stop.set()
                in_flight.release()
                break
    for _ in devs:
        q.put(None)
    q.join()
    for th in threads:
        th.join()
    keep = [i for i in range(N) if feats[i] is not None]
    np.savez(f"{out}/feats_raptor_{split}.npz", studies=np.array([ids[i] for i in keep]), feat=np.stack([feats[i] for i in keep]),
             logit=logits[keep], filled=filled[keep])
    print(f"saved feats_raptor_{split}.npz: {len(keep)} of {N} studies, feat {np.stack([feats[i] for i in keep[:1]]).shape[1:]}, "
          f"studies with no usable slice: {int((filled[keep] == 0).sum())} | {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    extract_features()
