# Solo member: our own single model, submitted on its own

Branch `solo-model`. Nothing here touches the existing kernels; every file is new.
`U` below is your Kaggle username. Run commands from the RSNA project folder.

| File | What it is |
|---|---|
| `kaggle/rsna-knee-solo/solo_model.py` | The model, shared by training and inference |
| `kaggle/rsna-knee-train-solo/train_solo.py` | Training job, one fold per process |
| `kaggle/rsna-knee-train-solo/build.py` | Builds a training notebook (two folds per T4 x2 session) |
| `kaggle/rsna-knee-solo/solo_driver.py` | Test studies -> `submission.csv`, our members only |
| `kaggle/rsna-knee-solo/build.py` | Builds the submission notebook for one cache contract |
| `research/oof_tools.py` | Merge / score / compare out-of-fold tables |
| `kaggle/rsna-knee-solo/smoke_local.py` | Local end-to-end test on synthetic DICOMs (CPU) |

## Before you start
- The training kernels read the 9-slice cache from a kernel output. Under your own account,
  first run `kaggle/rsna-knee-cache-9sl` once (CPU, about 1 h; change `id` in its
  `kernel-metadata.json` to `U/rsna-knee-cache-9sl`), or pass `--cache-kernel owner/rsna-knee-cache-9sl`
  if the owner has shared theirs with you.
- `RUN=../"Kaggle - 0Protocal"/skills/kaggle-research-to-run/scripts/kaggle_run.sh`

## 1. First standalone submission (existing E3 ConvNeXt-T, no training)
```bash
python kaggle/rsna-knee-solo/build.py --user U --contract e3 --dataset <owner>/rsna-knee-own-members --archs cnx
bash "$RUN" kaggle/rsna-knee-solo NvidiaTeslaT4
```
Read `_solo_receipt.json` in the output (studies covered, seconds per stage), then submit.
This gives the first real LB and runtime of a model of ours.

## 2. Check time and memory on a T4 (20 optimizer steps)
```bash
python kaggle/rsna-knee-train-solo/build.py --user U --suffix bench --folds 0 --bench
bash "$RUN" kaggle/rsna-knee-train-solo-bench NvidiaTeslaT4
```

## 3. Round 1: five folds on report labels
```bash
python kaggle/rsna-knee-train-solo/build.py --user U --suffix a --folds 0,1
python kaggle/rsna-knee-train-solo/build.py --user U --suffix b --folds 2,3
python kaggle/rsna-knee-train-solo/build.py --user U --suffix c --folds 4
for s in a b c; do OUTPUT_PATTERN='\.(json|csv)$' bash "$RUN" kaggle/rsna-knee-train-solo-$s NvidiaTeslaT4; done
python research/oof_tools.py merge runs/rsna-knee-train-solo-*/latest/res --out runs/solo/oof_round1.csv
python research/oof_tools.py score runs/solo/oof_round1.csv
```

## 4. Submit round 1 on its own
```bash
python kaggle/rsna-knee-solo/build.py --user U --contract 9sl --kernel U/rsna-knee-train-solo-a --kernel U/rsna-knee-train-solo-b --kernel U/rsna-knee-train-solo-c
bash "$RUN" kaggle/rsna-knee-solo NvidiaTeslaT4
```

## 5. Round 2: pseudo labels (out-of-fold predictions mixed into the report labels)
```bash
python kaggle/rsna-knee-train-solo/build.py --user U --suffix a2 --folds 0,1 --labels pseudo --pseudo runs/solo/oof_round1.csv
# same for b2 (2,3) and c2 (4); run; merge to runs/solo/oof_round2.csv; then:
python research/oof_tools.py diff runs/solo/oof_round1.csv runs/solo/oof_round2.csv
```
Keep round 2 only if `diff` says B is better (gain >= 0.003 and the 95% CI excludes 0).

## Local test
```bash
uv run --no-project --python 3.12 --with torch --with timm --with pydicom --with pandas --with scikit-learn python kaggle/rsna-knee-solo/smoke_local.py
```
