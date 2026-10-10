"""Local smoke test of the gpjw training path, on synthetic DICOMs and the CPU (plumbing, not accuracy).

  1. the cache job writes shards whose volumes equal the reader's own `load_series` output;
  2. the training cell indexes the shards, trains a tiny model for two folds/variants and writes
     checkpoints in the public format;
  3. the reader's own DICOM path (`infer.py` classes) reproduces the training job's gold predictions
     from the saved checkpoint, i.e. cache path == submission path.

Run: uv run --no-project --python 3.12 --with torch --with timm --with pydicom --with pandas \
       --with scikit-learn --with opencv-python-headless python kaggle/rsna-knee-gpjw/smoke_gpjw.py [--stage 1]
"""
import argparse, base64, json, os, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "rsna-knee-solo")]
import build_cache                      # noqa: E402
import smoke_local as corpus            # noqa: E402  (synthetic DICOM corpus of the solo smoke test)

T = corpus.T


def run(cmd, env):
    r = subprocess.run(cmd, env={**os.environ, **env}, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-4000:], r.stderr[-4000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd[:3]))}")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=3)
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="gpjw_smoke_"))
    comp = tmp / "input" / "competitions" / "rsna-knee-abnormality-detection"
    comp.mkdir(parents=True)
    studies, _, _, _ = corpus.make_corpus(comp)
    env = {"GPJW_INPUT": str(tmp / "input"), "GPJW_TMP": str(tmp / "t"), "GPJW_LOCAL": "1"}

    # ---- 1. cache job, two parts, several shards per part ----
    for part in (0, 1):
        cell = build_cache.cell_for(part, 2).replace("SHARD = 48", "SHARD = 6")
        (tmp / f"cache_p{part}.py").write_text(cell)
        out = run([sys.executable, str(tmp / f"cache_p{part}.py")],
                  {**env, "GPJW_WORK": str(tmp / "input" / f"rsna-knee-gpjw-cache-p{part}")})
        print(out.strip().splitlines()[0])
    sys.path.insert(0, str(tmp / "t" / "own_src"))
    from preprocess import load_series
    man = pd.concat(pd.read_csv(f, dtype={"study": str, "series": str})
                    for f in sorted((tmp / "input").glob("rsna-knee-gpjw-cache-p*/manifest_p*.csv")))
    series = pd.read_csv(comp / "train_series.csv", dtype=str)
    assert sorted(man.series) == sorted(series.SeriesInstanceUID), "manifest does not list every series once"
    assert man.error.isna().all() and (man.n == corpus.N_SLICE).all(), man[["n", "error"]].drop_duplicates()
    seen = 0
    for shard in sorted((tmp / "input").glob("rsna-knee-gpjw-cache-p*/cache384/*.npz")):
        z = np.load(shard)
        for key in z.files:
            row = series[series.SeriesInstanceUID == key].iloc[0]
            ref, _ = load_series(str(comp / "train_series" / row.StudyInstanceUID / key), row.Anatomical_Plane)
            assert z[key].dtype == np.uint8 and np.array_equal(z[key], ref), f"cached volume differs: {key}"
            seen += 1
    assert seen == len(series)
    n_shards = len(list((tmp / "input").glob("rsna-knee-gpjw-cache-p*/cache384/*.npz")))
    print(f"1. cache OK: {seen} series in {n_shards} shards equal load_series, manifest complete")
    if a.stage < 2:
        print("STAGE 1 PASSED"); return

    # ---- 2. training cell ----
    import build_train
    lab = pd.DataFrame(np.random.default_rng(1).random((len(studies), len(T))).round(3), index=pd.Index(studies, name="StudyInstanceUID"), columns=T)
    tr = pd.read_csv(comp / "train.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    is_gold = tr[T].notna().all(axis=1).values
    lab.insert(0, "is_gold", is_gold.astype(int))
    lab["fold"] = np.where(is_gold, -1, np.arange(len(studies)) % 5)
    for c in T:
        lab[f"gold_{c}"] = tr[c].values
    for c in T:                       # as in a mixed-label table: the held-out fold is scored on orig_*
        lab[f"orig_{c}"] = lab[c]
    lab.to_csv(tmp / "labels.csv")
    jobs = [{"tag": "ctrl", "fold": 3, "backbone": "convnext_atto", "res": 64, "extra": ["--no-pretrained", "--k", "2", "--k_eval", "3", "--epochs", "2", "--bs", "4", "--micro", "2", "--workers", "0", "--eval-every", "1", "--budget-h", "1"]},
            {"tag": "var", "fold": 3, "backbone": "convnext_atto", "res": 96, "extra": ["--no-pretrained", "--k", "2", "--k_eval", "3", "--epochs", "1", "--bs", "4", "--micro", "4", "--workers", "0", "--eval-every", "1"]}]
    cell = build_train.cell_for(jobs, tmp / "labels.csv", port_ref=None)
    (tmp / "train_cell.py").write_text(cell)
    out = run([sys.executable, str(tmp / "train_cell.py")], {**env, "GPJW_WORK": str(tmp / "work")})
    print("\n".join(l for l in out.splitlines() if l.startswith(("indexed", "job ", "timing", "ALL JOBS"))))
    res = tmp / "work" / "res"
    for j in jobs:
        for f in (f"{j['tag']}_fold3.pt", f"oof_{j['tag']}_fold3.csv", f"results_{j['tag']}_fold3.json", f"log_{j['tag']}_fold3.txt"):
            assert (res / f).is_file(), f"missing {f}"
    print("2. training OK: checkpoints, oof and results written for", [j["tag"] for j in jobs])

    # ---- 2b. fine-tuning from a checkpoint with per-cell loss weights ----
    import shutil
    (tmp / "input" / "earlier-run" / "res").mkdir(parents=True)
    shutil.copy(res / "ctrl_fold3.pt", tmp / "input" / "earlier-run" / "res" / "ctrl_fold3.pt")
    for c in T:
        lab[f"w_{c}"] = 0.3 if c == "Effusion" else 1.0
    lab.to_csv(tmp / "labels_w.csv")
    ft = [{"tag": "ft", "fold": 3, "backbone": "convnext_atto", "res": 64, "init": "ctrl_fold3.pt",
           "extra": ["--k", "2", "--k_eval", "3", "--epochs", "1", "--bs", "4", "--micro", "2", "--workers", "0", "--eval-every", "1", "--warm", "0.25", "--lr", "5e-5"]}]
    (tmp / "ft_cell.py").write_text(build_train.cell_for(ft, tmp / "labels_w.csv", port_ref=None))
    out = run([sys.executable, str(tmp / "ft_cell.py")], {**env, "GPJW_WORK": str(tmp / "work")})
    log = (res / "log_ft_fold3.txt").read_text()
    assert "initialised from ctrl_fold3.pt" in log and (res / "ft_fold3.pt").is_file(), log[-600:]
    print("2b. fine-tuning OK: started from ctrl_fold3.pt with weighted cells")

    # ---- 2c. feature job: head(saved features) is the checkpoint's own prediction ----
    fj = [{"tag": "ctrl", "fold": 3, "backbone": "convnext_atto", "res": 64, "init": "ctrl_fold3.pt", "script": "feats_gpjw.py",
           "extra": ["--k_eval", "3", "--workers", "0", "--bs", "3"]}]
    (tmp / "feats_cell.py").write_text(build_train.cell_for(fj, tmp / "labels.csv", port_ref=None))
    run([sys.executable, str(tmp / "feats_cell.py")], {**env, "GPJW_WORK": str(tmp / "work")})
    z = np.load(res / "feats_ctrl_fold3.npz")
    oof = pd.read_csv(res / "oof_ctrl_fold3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    st = [str(x) for x in z["studies"]]
    assert sorted(st) == sorted(studies) and z["feat"].shape[:3] == (len(st), 6, 3) and z["feat"].dtype == np.float16
    assert st[:int(is_gold.sum())] == lab.index[lab.is_gold == 1].tolist(), "gold studies are not first"
    rows = [st.index(u) for u in oof.index]
    diff = float(np.abs(1 / (1 + np.exp(-z["logit"][rows])) - oof[T].values).max())
    assert diff < 2e-3, f"features do not reproduce the checkpoint's predictions: {diff}"
    import torch
    from knee import KneeNet
    ck = torch.load(res / "ctrl_fold3.pt", map_location="cpu", weights_only=False)
    m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False)
    m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        again = m.head(torch.from_numpy(z["feat"].astype(np.float32)), torch.from_numpy(z["mask"]), torch.from_numpy(z["pos"])).numpy()
    assert float(np.abs(again - z["logit"]).max()) < 5e-3, "the head on the saved float16 features differs from the saved logits"
    print(f"2c. features OK: {z['feat'].shape}, head(features) == saved predictions (max abs difference {diff:.1e})")

    # ---- 3. the reader's DICOM path reproduces the training job's gold predictions ----
    import torch
    from infer import DicomStudyDataset, load_models
    from knee import build_study_table
    from preprocess import MAX_SLICES
    import infer
    real_load = infer.load_models.__globals__["torch"].load
    ck_path = str(res / "ctrl_fold3.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    from knee import KneeNet
    m = KneeNet(ck["args"]["backbone"].split(".")[0], pretrained=False, img_size=ck["args"]["res"])
    m.load_state_dict(ck["model"]); m.eval()
    gold = lab.index[lab.is_gold == 1].tolist()
    se = series[series.StudyInstanceUID.isin(gold)]
    root = str(comp / "train_series")
    n_files = {sr: min(len(os.listdir(f"{root}/{st}/{sr}")), MAX_SLICES) for st, sr in zip(se.StudyInstanceUID, se.SeriesInstanceUID)}
    ds = DicomStudyDataset(gold, build_study_table(se, n_files), root, 3)
    with torch.no_grad():
        p = torch.cat([m(*[t.unsqueeze(0) for t in ds[i]], res=ck["args"]["res"]).sigmoid() for i in range(len(ds))]).numpy()
    oof = pd.read_csv(res / "oof_ctrl_fold3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    ref = oof[oof.set == "gold"].loc[gold, T].values
    diff = float(np.abs(p - ref).max())
    assert diff < 1e-4, f"DICOM path and cache path disagree: {diff}"
    print(f"3. DICOM path == cache path: max abs difference {diff:.2e} on {len(gold)} gold studies")

    # ---- 4. head session: a head retrained on the saved features goes into a checkpoint the DICOM path reads ----
    import build_heads
    (tmp / "input" / "feats-run" / "res").mkdir(parents=True)
    shutil.copy(res / "feats_ctrl_fold3.npz", tmp / "input" / "feats-run" / "res" / "feats_ctrl_fold3.npz")
    cell = build_heads.cell_for([{"feats": "feats_ctrl_fold3.npz", "init": "ctrl_fold3.pt"}], tmp / "labels_w.csv", "smk", ["--epochs", "1", "--threads", "2"])
    (tmp / "heads_cell.py").write_text(cell)
    out = run([sys.executable, str(tmp / "heads_cell.py")], {**env, "GPJW_WORK": str(tmp / "work_heads")})
    hres = tmp / "work_heads" / "res"
    ck2 = torch.load(hres / "smk_fold3.pt", map_location="cpu", weights_only=False)
    assert ck2["args"]["head"]["backbone_of"] == "ctrl_fold3.pt" and ck2["args"]["backbone"] == ck["args"]["backbone"]
    assert all(torch.equal(ck2["model"][k], ck["model"][k]) for k in ck["model"] if k.startswith("enc.")), "the backbone changed"
    assert not torch.equal(ck2["model"]["cls_w"], ck["model"]["cls_w"]), "the head did not change"
    m2 = KneeNet(ck2["args"]["backbone"].split(".")[0], pretrained=False, img_size=ck2["args"]["res"])
    m2.load_state_dict(ck2["model"]); m2.eval()
    with torch.no_grad():
        p2 = torch.cat([m2(*[t.unsqueeze(0) for t in ds[i]], res=ck2["args"]["res"]).sigmoid() for i in range(len(ds))]).numpy()
    oof2 = pd.read_csv(hres / "oof_smk_fold3.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    diff = float(np.abs(p2 - oof2[oof2.set == "gold"].loc[gold, T].values).max())
    assert diff < 2e-3, f"the merged checkpoint through the DICOM path differs from the head's own predictions: {diff}"
    print(f"4. head session OK: {[l for l in out.splitlines() if l.startswith('gold-58')][0][:90]}; merged checkpoint through the DICOM path == head predictions ({diff:.1e})")
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
