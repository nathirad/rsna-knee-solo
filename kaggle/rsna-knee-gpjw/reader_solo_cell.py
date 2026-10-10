# The 2.5D reader alone on the test set (no stack): a fast submission used to compare our own
# checkpoints on the leaderboard's expert labels. Reader source = the public notebook's, unchanged.
import glob, json, os, subprocess, sys, time
import pandas as pd
OURS = __OURS__
t0 = time.time()


def find(name, prefix=False):
    for d, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        for f in sorted(files):
            if f == name or (prefix and f.startswith(name)):
                return os.path.join(d, f)
    raise FileNotFoundError(name)


target = "/kaggle/working/_own_env"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--force-reinstall",
                find("opencv_python_headless-4.12.0.88", prefix=True)], check=True)     # the build the training cache was made with
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", target,
                "--find-links", os.path.dirname(find("cnxt_v0_fold0.pt")),
                "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=False)
data = os.path.dirname(find("test_series.csv"))
ckpts = [find(n) for n in OURS]
env = {**os.environ, "PYTHONPATH": target + os.pathsep + os.environ.get("PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": "0"}
subprocess.run([sys.executable, "/kaggle/working/own_src/infer.py", data, "test", "/kaggle/working/_reader.csv"] + ckpts,
               check=True, env=env)
sample = pd.read_csv(f"{data}/sample_submission.csv", dtype={"StudyInstanceUID": str})
pred = pd.read_csv("/kaggle/working/_reader.csv", dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
labels = [c for c in sample.columns if c != "StudyInstanceUID"]
sub = pred.loc[sample.StudyInstanceUID, labels].reset_index()
assert sub[labels].notna().all().all() and len(sub) == len(sample)
sub.to_csv("/kaggle/working/submission.csv", index=False)
json.dump({"checkpoints": [os.path.basename(c) for c in ckpts], "studies": len(sub), "seconds": time.time() - t0},
          open("/kaggle/working/_reader_receipt.json", "w"))
print(f"reader alone: {len(ckpts)} checkpoints {[os.path.basename(c) for c in ckpts]}, {len(sub)} studies, {time.time() - t0:.0f}s")
