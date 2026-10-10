"""Local end-to-end smoke test of the solo pipeline, on synthetic DICOMs and the CPU.

It checks the plumbing, not the accuracy:
  1. the training-cache script and the submission script decode the SAME pixels for the
     same study (the cache contract), at the patched resolution / slice count / band;
  2. train_solo.py runs for both window modes and for the pseudo-label round, and writes
     member_*.json, oof_*.csv, gold_*.csv and a checkpoint;
  3. the submission script reproduces the training job's own predictions for the gold
     studies from the saved checkpoint, and writes a complete submission.csv;
  4. an E3-style members.json (plain list, no window embedding) still loads.

Run from anywhere:
  uv run --python 3.12 --with torch --with timm --with pydicom --with pandas \
         --with scikit-learn python kaggle/rsna-knee-solo/smoke_local.py
"""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid

HERE = Path(__file__).resolve().parent
KAGGLE = HERE.parent
sys.path.insert(0, str(HERE))
import build as solo_build   # noqa: E402

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
PLANES = {  # plane -> (row direction, column direction, axis the stack advances along)
    "Sagittal": ([0, 1, 0, 0, 0, -1], 0), "Coronal": ([1, 0, 0, 0, 0, -1], 1),
    "Axial": ([1, 0, 0, 0, 1, 0], 2)}
SERIES = [("SAG PD FS", "Sagittal", 3000, 30, "FS", 1, 1), ("COR T1", "Coronal", 500, 10, "", 0, 0),
          ("AX PD FS", "Axial", 3200, 35, "FS", 1, 1)]
N_TRAIN, N_GOLD, N_SLICE, PX = 40, 6, 11, 48
rng = np.random.default_rng(0)


def write_series(folder, study, series, desc, plane, tr, te, opts, side, with_tag):
    folder.mkdir(parents=True)
    iop, axis = PLANES[plane]
    base = np.array([80.0 if side == "L" else -80.0, -60.0, 60.0])
    blob = rng.integers(0, 600, size=(N_SLICE, PX, PX)).astype(np.uint16)
    for k in rng.permutation(N_SLICE):          # written out of order: file names carry no order
        ds = Dataset()
        ds.file_meta = FileMetaDataset()
        ds.file_meta.MediaStorageSOPClassUID = MRImageStorage
        ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.SOPClassUID, ds.SOPInstanceUID = MRImageStorage, ds.file_meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.Modality = study, series, "MR"
        ds.SeriesDescription, ds.RepetitionTime, ds.EchoTime = desc, tr, te
        ds.ScanningSequence, ds.ScanOptions = "SE", opts
        if with_tag:
            ds.Laterality = side
        pos = base.copy(); pos[axis] += 4.0 * k
        ds.ImagePositionPatient, ds.ImageOrientationPatient = [float(v) for v in pos], iop
        ds.PixelSpacing, ds.InstanceNumber = [3.5, 3.5], int(k) + 1
        ds.Rows, ds.Columns, ds.SamplesPerPixel, ds.PhotometricInterpretation = PX, PX, 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
        ds.PixelData = (blob[k] + 40 * k).astype(np.uint16).tobytes()
        ds.save_as(folder / f"{ds.SOPInstanceUID}.dcm", enforce_file_format=True)


def make_corpus(root):
    studies, srows = [f"1.2.826.0.1.{1000 + i}" for i in range(N_TRAIN)], []
    for i, st in enumerate(studies):
        side = "R" if i % 2 else "L"
        for j, (desc, plane, tr, te, opts, fluid, fs) in enumerate(SERIES):
            if j == 2 and i % 4 == 0:           # a quarter of the studies lack the axial series
                continue
            se = f"{st}.{j + 1}"
            write_series(root / "train_series" / st / se, st, se, desc, plane, tr, te, opts, side, i % 3 == 0)
            srows.append({"StudyInstanceUID": st, "SeriesInstanceUID": se, "Fluid_Sensitive": fluid,
                          "Fat_Suppression": fs, "Anatomical_Plane": plane})
    series = pd.DataFrame(srows)
    series.to_csv(root / "train_series.csv", index=False)
    lab = pd.DataFrame(rng.random((N_TRAIN, len(T))), columns=T)
    train = pd.DataFrame({"StudyInstanceUID": studies,
                          "Report": [f"report text {i // 2}" for i in range(N_TRAIN)]})   # duplicated reports
    for t in T:
        train[t] = np.where(np.arange(N_TRAIN) < N_GOLD, (lab[t] > 0.5).astype(float), np.nan)
    for t in T:                                 # every gold column needs both classes
        train.loc[0, t], train.loc[1, t] = 1.0, 0.0
    train.to_csv(root / "train.csv", index=False)
    # test = the gold studies plus four others, copied: same UIDs, so pixels can be compared
    test = studies[:N_GOLD + 4]
    for st in test:
        shutil.copytree(root / "train_series" / st, root / "test_series" / st)
    pd.DataFrame({"StudyInstanceUID": test + ["9.9.9.no-series"]}).to_csv(root / "test.csv", index=False)
    series[series.StudyInstanceUID.isin(test)].to_csv(root / "test_series.csv", index=False)
    pilk = pd.DataFrame({"StudyInstanceUID": studies})
    for t in T:
        pilk[t], pilk[t + "__conf"] = lab[t].round(3), rng.random(N_TRAIN).round(3)
    sl4 = pd.DataFrame({"StudyInstanceUID": studies, **{t: np.where(lab[t] > 0.5, 0.95, 0.05) for t in T}})
    return studies, test, pilk, sl4


def run(cmd, cwd, env=None):
    r = subprocess.run(cmd, cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAILED: {' '.join(map(str, cmd[:3]))}")
    return r.stdout


def solo_script(contract, tmp, name):
    script = "\n".join([solo_build.preproc(contract), "# %%\n" + (HERE / "solo_model.py").read_text(),
                        (HERE / "solo_driver.py").read_text()])
    p = tmp / name
    p.write_text(script)
    return p


def main():
    tmp = Path(tempfile.mkdtemp(prefix="solo_smoke_"))
    root = tmp / "corpus"; root.mkdir()
    studies, test, pilk, sl4 = make_corpus(root)
    print(f"corpus: {N_TRAIN} train studies ({N_GOLD} gold), {len(test)} test studies, in {tmp}")

    # ---- 1. training cache, with the 9sl cache script at 64 px -------------------------
    cache_src = (KAGGLE / "rsna-knee-cache-9sl" / "rsna-knee-cache-9sl.py").read_text()
    assert cache_src.count("CACHE_IMG = 256\n") == 1
    # As a notebook each cell compiles on its own; as one script the __future__ import of a
    # later cell is no longer first, so it is dropped here (Python 3.10+ does not need it).
    (tmp / "cache_job.py").write_text(cache_src.replace("CACHE_IMG = 256\n", "CACHE_IMG = 64\n")
                                      .replace("from __future__ import annotations\n", ""))
    run([sys.executable, str(tmp / "cache_job.py")], root)
    cdir = root / "cache_out"
    meta = json.loads((cdir / "cache_meta.json").read_text())
    assert (meta["img"], meta["slices"], meta["band"]) == (64, 9, [0.15, 0.85]), meta
    print(f"1. training cache built: {meta['img']} px, {meta['slices']} slices, band {meta['band']}")

    # ---- 2. training, both window modes, then the pseudo-label round -------------------
    ind = tmp / "in"; ind.mkdir()
    shutil.copy(root / "train.csv", ind / "train.csv")
    pilk.to_csv(ind / "report_labels_v2.csv", index=False)
    sl4.to_csv(ind / "llm_labels_v4_blend.csv", index=False)
    base = {"IN_DIR": str(ind), "CACHE_DIR": str(cdir), "ARCH": "timm:resnet18", "IMG": "64",
            "EPOCHS": "2", "EVAL_EVERY": "1", "PRETRAINED": "0", "BATCH_STUDIES": "4", "ACCUM": "2"}
    job = str(KAGGLE / "rsna-knee-train-solo" / "train_solo.py")
    for mode in ("all", "one"):
        for fold in range(5):
            run([sys.executable, job], tmp, {**base, "WINDOWS": mode, "FOLD": str(fold),
                                             "RES_DIR": str(tmp / f"res_{mode}")})
    tools = str(KAGGLE.parent / "research" / "oof_tools.py")
    oof1 = tmp / "oof_round1.csv"
    print("  " + run([sys.executable, tools, "merge", str(tmp / "res_all"), "--out", str(oof1)], tmp).strip())
    n_oof = len(pd.read_csv(oof1))
    assert n_oof == N_TRAIN - N_GOLD, f"out-of-fold table has {n_oof} rows"
    run([sys.executable, job], tmp, {**base, "WINDOWS": "all", "FOLD": "0", "LABELS": "pseudo",
                                     "PSEUDO_CSV": str(oof1), "RES_DIR": str(tmp / "res_pseudo")})
    assert (tmp / "res_pseudo" / "member_timm-resnet18_64_pseudo_all_f0_s0.json").is_file()
    run([sys.executable, job], tmp, {**base, "WINDOWS": "one", "FOLD": "0", "LABELS": "pseudo_keep",
                                     "PSEUDO_CSV": str(oof1), "RES_DIR": str(tmp / "res_pseudo")})
    assert (tmp / "res_pseudo" / "member_timm-resnet18_64_pseudo_keep_one_f0_s0.json").is_file()
    # a table from another split must be refused
    bad = pd.read_csv(oof1); bad["fold"] = (bad["fold"] + 1) % 5; bad.to_csv(tmp / "oof_bad.csv", index=False)
    r = subprocess.run([sys.executable, job], cwd=tmp, capture_output=True, text=True,
                       env={**os.environ, **base, "WINDOWS": "all", "FOLD": "0", "LABELS": "pseudo",
                            "PSEUDO_CSV": str(tmp / "oof_bad.csv"), "RES_DIR": str(tmp / "res_bad")})
    assert r.returncode != 0 and "fold differs" in r.stderr, "a wrong-split pseudo table was accepted"
    print(f"2. trained 5 folds x 2 window modes + pseudo round; out-of-fold table {n_oof} rows; "
          f"wrong-split pseudo table refused")

    # ---- 3. submission script against the same studies ---------------------------------
    solo_build.CONTRACTS["smoke"] = {"img": 64, "groups": 3, "band": (0.15, 0.85)}
    script = solo_script("smoke", tmp, "_solo_smoke.py")
    C = np.load(cdir / "cache_64_part00.npy"); Mk = np.load(cdir / "mask.npy")
    for mode in ("all", "one"):
        out = tmp / f"sub_{mode}.csv"
        run([sys.executable, str(script)], root, {"SOLO_ASSETS": str(tmp / f"res_{mode}"), "SOLO_OUT": str(out),
                                                  "SOLO_DUMP": str(tmp / f"dump_{mode}.npz")})
        sub = pd.read_csv(out, dtype={"StudyInstanceUID": str})
        assert sub.columns.tolist() == ["StudyInstanceUID", *T] and len(sub) == len(test) + 1
        assert sub[T].notna().all().all()
        assert (sub.loc[sub.StudyInstanceUID == "9.9.9.no-series", T] == 0.5).all().all()
        d = np.load(tmp / f"dump_{mode}.npz", allow_pickle=True)
        idx = [meta["studies"].index(s) for s in d["studies"]]
        assert np.array_equal(d["cache"], C[idx]), "submission pixels differ from the training cache"
        assert np.array_equal(d["mask"], Mk[idx]), "slot masks differ"
        worst = 0.0
        for k, name in enumerate(d["members"]):      # checkpoint (fp16) vs the job's own gold predictions
            g = pd.read_csv(tmp / f"res_{mode}" / name.replace("model_", "gold_").replace(".pt", ".csv"),
                            dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
            rows = [list(d["studies"]).index(s) for s in g.index]
            worst = max(worst, float(np.abs(d["probs"][k][rows] - g[T].values).max()))
        assert worst < 1e-2, f"predictions differ from training by {worst}"
        rec = json.loads((tmp / "_solo_receipt.json").read_text())
        print(f"3. mode {mode}: pixels identical to the training cache for {len(idx)} studies; "
              f"5 members reproduce training predictions (max diff {worst:.1e}); "
              f"submission {sub.shape}, {rec['n_without_prediction']} study without series -> 0.5")

    # ---- 4. an E3-style members.json (plain list, weights without win_emb) -------------
    import torch
    e3 = tmp / "e3_assets"; e3.mkdir()
    sd = torch.load(tmp / "res_one" / "model_timm-resnet18_64_pilk_one_f0_s0.pt")
    sd.pop("head.win_emb")
    torch.save(sd, e3 / "model_legacy.pt")
    (e3 / "members.json").write_text(json.dumps([
        {"file": "model_legacy.pt", "arch": "timm:resnet18", "img": 224},
        {"file": "model_dino.pt", "arch": "dino", "img": 224}]))
    run([sys.executable, str(solo_script("e3", tmp, "_solo_e3.py"))], root,
        {"SOLO_ASSETS": str(e3), "SOLO_OUT": str(tmp / "sub_e3.csv"), "SOLO_ARCHS": "timm:resnet18"})
    sub = pd.read_csv(tmp / "sub_e3.csv")
    assert len(sub) == len(test) + 1 and sub[T].notna().all().all()
    print("4. E3-style members.json loads under the e3 contract (336 px, 3 slices); DINO entry filtered out")
    shutil.rmtree(tmp)
    print("ALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
