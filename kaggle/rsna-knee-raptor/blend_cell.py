# Ours: run our 2.5D reader and rank-blend it into the single model's submission.
# The cell above (the public notebook, unchanged) has written /kaggle/working/submission.csv.
# If anything here fails, that file is kept as it is.
import glob as _b_glob, json as _b_json, os as _b_os, shutil as _b_shutil, subprocess as _b_sp, sys as _b_sys, time as _b_time
import pandas as _b_pd
READER_W = __WEIGHT__      # the reader's weight; fixed before submitting, not tuned on the leaderboard
READER_W_FINDING = __WEIGHT_FINDING__      # findings with a weight of their own
_b_ours = __OURS__
_b_pub, _b_bak = "/kaggle/working/submission.csv", "/kaggle/working/_single_model.csv"
_b_shutil.copy(_b_pub, _b_bak)
_b_t0 = _b_time.time()
try:
    def _b_find(name, prefix=False):
        for d, dirs, files in _b_os.walk("/kaggle/input"):
            dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
            for f in sorted(files):
                if f == name or (prefix and f.startswith(name)):
                    return _b_os.path.join(d, f)
        raise FileNotFoundError(name)

    # the OpenCV build the reader's training cache was made with
    _b_sp.run([_b_sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall",
               _b_find("opencv_python_headless-4.12.0.88", prefix=True)], check=True)
    _b_data = _b_os.path.dirname(_b_find("test_series.csv"))
    _b_ck = [_b_find(n) for n in _b_ours]
    _b_env = dict(_b_os.environ)
    try:
        import gc as _b_gc, torch as _b_torch
        _b_gc.collect(); _b_torch.cuda.empty_cache()
        _b_free = [_b_torch.cuda.mem_get_info(i)[0] for i in range(_b_torch.cuda.device_count())]
        _b_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_b_free)), key=lambda i: _b_free[i]))
        print("reader: free GPU memory (GB)", [round(f / 1e9, 1) for f in _b_free], flush=True)
    except Exception as _b_e:
        print("reader: GPU pick skipped:", repr(_b_e))
    _b_sp.run([_b_sys.executable, "/kaggle/working/own_src/infer.py", _b_data, "test", "/kaggle/working/_reader.csv"] + _b_ck,
              check=True, env=_b_env, timeout=6 * 3600)
    _b_a = _b_pd.read_csv(_b_bak, dtype={"StudyInstanceUID": str})
    _b_r = _b_pd.read_csv("/kaggle/working/_reader.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[_b_a.StudyInstanceUID].reset_index()
    _b_lab = [c for c in _b_a.columns if c != "StudyInstanceUID"]
    assert _b_r[_b_lab].notna().all().all(), "reader predictions contain NaN"
    assert (_b_r[_b_lab].nunique() > 1).all(), "reader predictions are constant"
    _b_rank = lambda d: d[_b_lab].rank(method="average", pct=True)
    assert set(READER_W_FINDING) <= set(_b_lab), f"unknown finding in {READER_W_FINDING}"
    _b_w = _b_pd.Series({c: float(READER_W_FINDING.get(c, READER_W)) for c in _b_lab})
    _b_final = _b_a.copy()
    _b_final[_b_lab] = ((1 - _b_w) * _b_rank(_b_a) + _b_w * _b_rank(_b_r)).rank(method="average", pct=True)
    _b_final.to_csv(_b_pub, index=False)
    _b_json.dump({"checkpoints": [_b_os.path.basename(p) for p in _b_ck], "weight": READER_W, "weight_finding": READER_W_FINDING, "studies": len(_b_final),
                  "reader_seconds": _b_time.time() - _b_t0,
                  "mean_rank_corr": float(_b_rank(_b_a).corrwith(_b_rank(_b_r)).mean())}, open("/kaggle/working/_blend_receipt.json", "w"))
    print(f"reader: blended {len(_b_ck)} checkpoints at weight {READER_W}, per finding {READER_W_FINDING}, in {_b_time.time() - _b_t0:.0f}s", flush=True)
except Exception as _b_e:
    _b_shutil.copy(_b_bak, _b_pub)
    print("reader FAILED, the single model's submission is kept:", repr(_b_e))
