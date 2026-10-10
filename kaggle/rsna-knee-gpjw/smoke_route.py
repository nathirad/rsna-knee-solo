"""Local check of the last cell of a stack + reader notebook built with --route (no GPU, no DICOM).

The cell runs against a stand-in /kaggle tree: a stack submission, a stand-in reader script and a stand-in
for the older model's script (the real script is checked on Kaggle by rsna-knee-solo-check and locally by
kaggle/rsna-knee-solo/smoke_local.py). Cases: routed; the older model fails; too little time; a member is
missing. Every case must write the blend. Routed: the named findings are the rank blend of the stack with
the older model, the others with the reader. All other cases: exactly the notebook without --route.

Usage: python smoke_route.py [../rsna-knee-stack-route/rsna-knee-stack-route.ipynb]
"""
import base64, contextlib, io, json, re, shutil, sys, tempfile, time
from pathlib import Path

import numpy as np
import pandas as pd

T = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ROUTE = ["Effusion", "Synovitis"]
NB = Path(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else Path(__file__).parent.parent / "rsna-knee-stack-route/rsna-knee-stack-route.ipynb"
cell = "".join([c for c in json.loads(NB.read_text())["cells"] if c["cell_type"] == "code"][-1]["source"])
assert cell.count("b64decode(") == 1 and "_r_cols" in cell, "this notebook was not built with --route"

# stand-ins: both write values that depend only on the study id, so the expected blend can be computed here
VALUES = '''
import hashlib, numpy as np, pandas as pd
T = %r
def table(ids, salt):
    rows = [[int(hashlib.md5(f"{salt}|{s}|{t}".encode()).hexdigest()[:8], 16) / 16 ** 8 for t in T] for s in ids]
    return pd.concat([pd.DataFrame({"StudyInstanceUID": list(ids)}), pd.DataFrame(rows, columns=T)], axis=1)
''' % (T,)
READER = VALUES + '''
import sys
ids = pd.read_csv(sys.argv[1] + "/test.csv", dtype=str).StudyInstanceUID
table(ids, "reader").to_csv(sys.argv[3], index=False)
'''
SOLO = VALUES + '''
import json, os, sys
from pathlib import Path
if os.environ.get("SMOKE_SOLO_FAILS"):
    sys.exit(3)
assert os.environ["SLOT_SCHEME"] == "recovered" and os.environ["SOLO_KEEP"] == "mode=one,labels=pilk"
found = [p for p in sorted(Path(os.environ["SOLO_ASSETS"]).rglob("member_*.json")) if "_one_" in p.name]   # what SOLO_KEEP mode=one keeps
out = Path(os.environ["SOLO_OUT"])
ids = pd.read_csv(Path(os.environ["SMOKE_DATA"]) / "test.csv", dtype=str).StudyInstanceUID
table(ids, "solo").to_csv(out, index=False)
out.with_name("_solo_receipt.json").write_text(json.dumps({"members": [p.name for p in found], "n_without_prediction": 0}))
'''
exec(VALUES)
rank = lambda d: d[T].rank(method="average", pct=True)


def run(case, hours_used=1.0, solo_fails=False, drop_member=False):
    root = Path(tempfile.mkdtemp(prefix="route_smoke_"))
    work, inp = root / "working", root / "input"
    ids = [f"1.2.{i}" for i in range(40)]
    (inp / "comp").mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": ids}).to_csv(inp / "comp/test.csv", index=False)
    (inp / "comp/test_series.csv").write_text("StudyInstanceUID,SeriesInstanceUID\n")
    (inp / "reader").mkdir()
    for k in range(3):
        (inp / f"reader/cnxt_v0_fold{k}.pt").write_bytes(b"")
    for slug, files in {"rsna-knee-gpjw-train-r1": ["ctrl_fold3.pt"], "rsna-knee-gpjw-train-r2": ["ctrl_fold4.pt"],
                        "rsna-knee-train-solo-a": ["member_cnx_224_pilk_one_f1_s0.json", "member_cnx_224_pilk_one_f2_s0.json"],
                        "rsna-knee-train-solo-b": ["member_cnx_224_pilk_one_f3_s0.json", "member_cnx_224_pilk_one_f4_s0.json"],
                        "rsna-knee-train-solo-ab": ["member_cnx_224_pilk_one_f0_s0.json", "member_cnx_224_pilk_all_f0_s0.json"]}.items():
        d = inp / "notebooks/someone" / slug / "res"
        d.mkdir(parents=True)
        for f in files[1:] if drop_member and slug.endswith("solo-a") else files:
            (d / f).write_text("{}")
    (work / "own_src").mkdir(parents=True)
    (work / "own_src/infer.py").write_text(READER)
    stack = table(ids, "stack")
    stack.to_csv(work / "submission.csv", index=False)
    (root / "meminfo").write_text("MemTotal:       31000000 kB\nMemAvailable:   20000000 kB\n")
    src = re.sub(r'b64decode\("[A-Za-z0-9+/=]+"\)', f'b64decode("{base64.b64encode(SOLO.encode()).decode()}")', cell)
    src = src.replace("/kaggle/working", str(work)).replace("/kaggle/input", str(inp)).replace("/proc/meminfo", str(root / "meminfo"))
    import os
    os.environ["SMOKE_DATA"] = str(inp / "comp")
    os.environ.pop("SMOKE_SOLO_FAILS", None)
    if solo_fails:
        os.environ["SMOKE_SOLO_FAILS"] = "1"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, "<last cell>", "exec"), {"T0": time.time() - hours_used * 3600})
    out = buf.getvalue()
    assert "own reader FAILED" not in out, out
    if "--show" in sys.argv:
        print(out)
    got = pd.read_csv(work / "submission.csv", dtype={"StudyInstanceUID": str})
    rec = json.loads((work / "_own_receipt.json").read_text())
    reader, solo = table(ids, "reader"), table(ids, "solo")
    plain = ((1 - 0.30) * rank(stack) + 0.30 * rank(reader)).rank(method="average", pct=True)
    leg = reader.copy()
    leg[ROUTE] = solo[ROUTE]
    routed = ((1 - 0.30) * rank(stack) + 0.30 * rank(leg)).rank(method="average", pct=True)
    assert got.StudyInstanceUID.tolist() == ids and len(rec["checkpoints"]) == 5
    want = routed if case == "routed" else plain
    assert np.allclose(got[T].values, want.values), f"{case}: the blend is not the expected one"
    assert (rec["route"]["note"] == "routed") == (case == "routed"), rec["route"]
    assert case != "routed" or not np.allclose(routed[ROUTE].values, plain[ROUTE].values)
    note = [l for l in out.splitlines() if "from the older model" in l][-1]
    print(f"ok  {case:18s} {note[:170]}")
    shutil.rmtree(root)


run("routed")
run("older model fails", solo_fails=True)
run("too little time", hours_used=8.3)
run("member missing", drop_member=True)
print("ALL ROUTE CHECKS PASSED")
