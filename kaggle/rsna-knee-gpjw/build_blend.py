"""Build rsna-knee-stack-reader5: the public notebook "Stack + 2.5D ConvNeXt MIL" (LB 0.944 with its
three reader checkpoints) with our own fold checkpoints of the same recipe added to the reader.

Every cell is the public notebook's, unchanged, except the last code cell: its checkpoint list gets
our files appended, the reader gets a time limit (what is left of 8.6 h) and a small receipt is
written. The blend weight stays 0.30. If the reader fails
for any reason the notebook keeps the stack's submission, as in the original.

--route takes the named findings of the reader leg from our older 5-fold model instead (kaggle/rsna-knee-solo/,
run as a script after the reader, with its own time and memory check). If that model fails for any reason the
reader's own columns are used, which is the notebook without --route.

Usage: python build_blend.py --user <kaggle-user> --ours ctrl_fold3.pt ctrl_fold4.pt --from-kernels rsna-knee-gpjw-train-r1 rsna-knee-gpjw-train-r2
       ... --slug rsna-knee-stack-reader5-route --route Effusion Synovitis --solo-kernels rsna-knee-train-solo-a rsna-knee-train-solo-b rsna-knee-train-solo-ab
"""
import argparse, ast, base64, importlib.util, json
from pathlib import Path

HERE = Path(__file__).parent
RSNA = HERE.parent.parent
NB = "rsna-knee-stack-2-5d-convnext-mil-lb-0-944"

ap = argparse.ArgumentParser()
ap.add_argument("--user", required=True)
ap.add_argument("--ours", nargs="+", required=True)
ap.add_argument("--from-kernels", nargs="+", required=True)
ap.add_argument("--slug", default="rsna-knee-stack-reader5")
ap.add_argument("--public-folds", nargs="*", type=int, default=None, help="keep only these public fold checkpoints (default: all three)")
ap.add_argument("--route", nargs="*", default=[], help="findings of the reader leg to take from our older 5-fold model")
ap.add_argument("--solo-kernels", nargs="*", default=[], help="training kernels whose output holds that model's five members")
a = ap.parse_args()
assert bool(a.route) == bool(a.solo_kernels), "--route and --solo-kernels go together"
src = RSNA / f"research/public_notebooks/knee_g3/{NB}"
nb = json.loads((src / f"{NB}.ipynb").read_text())
code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"]
i = code[-1]
cell = "".join(nb["cells"][i]["source"])
old = '    _o_ck = sorted(_o_glob.glob(_o_assets + "/cnxt_v0_fold*.pt"))\n'
new = old + (
    "    # our own folds of the same recipe, trained on our account; all of them must be found\n"
    f"    _o_ours = {json.dumps(a.ours)}\n"
    '    _o_extra = [p for n in _o_ours for pat in ("/kaggle/input/*/res/", "/kaggle/input/*/*/res/", "/kaggle/input/*/*/*/res/")\n'
    "                for p in sorted(_o_glob.glob(pat + n))]\n"
    '    assert len(_o_extra) == len(_o_ours), f"our checkpoints: found {_o_extra}"\n'
    "    _o_ck = _o_ck + _o_extra\n")
assert cell.count(old) == 1, "the public cell changed: the checkpoint line was not found once"
cell = cell.replace(old, new)
if a.public_folds is not None:      # our fine-tuned versions replace the other public folds
    keep = "    _o_ck = [p for p in _o_ck if any(p.endswith(f\"_fold{k}.pt\") for k in %s)]\n" % a.public_folds
    cell = cell.replace(old, old + keep)
old = '    print(f"own reader: blended {len(_o_ck)} checkpoints at weight {OWN_W} in {_o_time.time() - _o_t0:.0f}s")\n'
new = old + (
    "    import json as _o_json\n"
    '    _o_json.dump({"checkpoints": [_o_os.path.basename(p) for p in _o_ck], "weight": OWN_W, "studies": len(_o_final),\n'
    '                  "reader_seconds": _o_time.time() - _o_t0}, open("/kaggle/working/_own_receipt.json", "w"))\n')
assert cell.count(old) == 1
cell = cell.replace(old, new)
# time guard: the public cell has none, and a notebook that passes 9 h is lost, not scored as the stack
old = '              check=True, env=_o_env)\n'
new = '              check=True, env=_o_env, timeout=_o_left)\n'
assert cell.count(old) == 1
cell = cell.replace(old, new)
old = '    _o_t0 = _o_time.time()\n'
new = ('    _o_used = _o_time.time() - globals().get("T0", _o_time.time())   # seconds since the stack started\n'
       '    _o_left = 8.6 * 3600 - _o_used                                     # the notebook may run 9 h in all\n'
       '    assert _o_left > 900, f"no time left for the reader ({_o_used / 3600:.2f} h used)"\n') + old
assert cell.count(old) == 1
cell = cell.replace(old, new)
cell = cell.replace('"reader_seconds": _o_time.time() - _o_t0}', '"reader_seconds": _o_time.time() - _o_t0, "stack_seconds": _o_used}')
if a.route:
    # the older model's submission script, as kaggle/rsna-knee-solo/build.py assembles it (checked on Kaggle by rsna-knee-solo-check)
    spec = importlib.util.spec_from_file_location("solo_build", RSNA / "kaggle/rsna-knee-solo/build.py")
    solo_build = importlib.util.module_from_spec(spec); spec.loader.exec_module(solo_build)
    solo_dir = RSNA / "kaggle/rsna-knee-solo"
    script = "\n".join([solo_build.preproc("9sl"), "# %%\n" + (solo_dir / "solo_model.py").read_text(), (solo_dir / "solo_driver.py").read_text()])
    ast.parse(script)
    old = '    _o_rank = lambda d: d[_o_labels].rank(method="average", pct=True)\n'
    new = (
        "    # ---- per-finding source (ours): these columns of the reader leg come from our older 5-fold model ----\n"
        "    # ConvNeXt-T 224 px on a 9-slice cache, trained with a low loss weight on report-silent cells. On the 58\n"
        "    # expert-labelled studies it reads these findings better than the reader (Effusion 0.970 vs 0.877,\n"
        "    # Synovitis 0.802 vs 0.762). If it fails or time or memory is short, the reader's own columns stay.\n"
        f"    _r_cols, _r_note, _r_t0 = {json.dumps(a.route)}, \"not run\", _o_time.time()\n"
        "    try:\n"
        "        import base64 as _r_b64, json as _r_js\n"
        '        _r_dir = "/kaggle/working/_route"\n'
        "        _o_os.makedirs(_r_dir, exist_ok=True)\n"
        '        with open(_r_dir + "/_solo.py", "wb") as _r_f:\n'
        f'            _r_f.write(_r_b64.b64decode("{base64.b64encode(script.encode()).decode()}"))\n'
        '        _r_mem = [p for pat in ("/kaggle/input/*/res/", "/kaggle/input/*/*/res/", "/kaggle/input/*/*/*/res/")\n'
        '                  for p in sorted(_o_glob.glob(pat + "member_cnx_224_pilk_one_f*_s0.json"))]\n'
        '        assert len(_r_mem) == 5, f"older model: found {_r_mem}"\n'
        '        _r_left = 8.7 * 3600 - (_o_time.time() - globals().get("T0", _o_time.time()))\n'
        '        assert _r_left > 2400, f"only {_r_left / 60:.0f} min left"\n'
        '        with open("/proc/meminfo") as _r_f:\n'
        '            _r_gb = {l.split(":")[0]: int(l.split()[1]) / 1024 ** 2 for l in _r_f if ":" in l}["MemAvailable"]\n'
        '        assert _r_gb > 10, f"only {_r_gb:.1f} GB of memory available"\n'
        '        _r_env = {k: v for k, v in _o_os.environ.items() if not k.startswith("SOLO_") and k != "RSNA_ORDER_CACHE"}\n'
        '        _r_env.update(SOLO_OUT=_r_dir + "/solo.csv", SOLO_ASSETS=_o_os.path.commonpath(_r_mem),\n'
        '                      SOLO_KEEP="mode=one,labels=pilk", SLOT_SCHEME="recovered")\n'
        "        try:\n"
        "            _r_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]\n"
        '            _r_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(len(_r_free)) if _r_free[i] > 8e9 or _r_free[i] == max(_r_free))\n'
        "        except Exception as _r_e:\n"
        '            print("older model: GPU pick skipped:", repr(_r_e))\n'
        '        print(f"older model: {_r_left / 60:.0f} min left, {_r_gb:.1f} GB free, GPUs {_r_env.get(\'CUDA_VISIBLE_DEVICES\')}", flush=True)\n'
        '        _o_sp.run([_o_sys.executable, "-u", _r_dir + "/_solo.py"], check=True, env=_r_env, timeout=_r_left - 240)\n'
        '        _r_rec = _r_js.load(open(_r_dir + "/_solo_receipt.json"))\n'
        '        _r_solo = _o_pd.read_csv(_r_dir + "/solo.csv", dtype={"StudyInstanceUID": str})\n'
        '        _r_solo = _r_solo.set_index("StudyInstanceUID").loc[_o_stack.StudyInstanceUID].reset_index()\n'
        '        assert len(_r_rec["members"]) == 5 and _r_rec["n_without_prediction"] <= 0.05 * len(_r_solo), f"older model: {_r_rec}"\n'
        '        assert _r_solo[_r_cols].notna().all().all() and (_r_solo[_r_cols].nunique() > 1).all(), "older model: unusable columns"\n'
        "        _o_own[_r_cols] = _r_solo[_r_cols].values\n"
        '        _r_note = "routed"\n'
        "    except Exception as _r_e:\n"
        '        _r_note = "kept the reader\'s columns: " + repr(_r_e)\n'
        '    print(f"own reader: {_r_cols} from the older model -> {_r_note} ({_o_time.time() - _r_t0:.0f}s)", flush=True)\n') + old
    assert cell.count(old) == 1
    cell = cell.replace(old, new)
    old = '"stack_seconds": _o_used}'
    assert cell.count(old) == 1
    cell = cell.replace(old, '"stack_seconds": _o_used,\n                  "route": {"findings": _r_cols, "note": _r_note, "seconds": _o_time.time() - _r_t0}}')
ast.parse(cell)
nb["cells"][i]["source"] = cell
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None
out = HERE.parent / a.slug
out.mkdir(exist_ok=True)
(out / f"{a.slug}.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((src / "kernel-metadata.json").read_text())
meta.pop("id_no", None)          # the public notebook's own id: with it the push would target their notebook (refused, 403)
meta.update(id=f"{a.user}/{a.slug}", title=a.slug, code_file=f"{a.slug}.ipynb", is_private="true",
            kernel_sources=meta["kernel_sources"] + [f"{a.user}/{k}" for k in a.from_kernels + a.solo_kernels])
(out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
print(f"built {a.slug}/: {len(nb['cells'])} cells, reader = public folds + {a.ours}" + (f", {a.route} from the older model" if a.route else "")
      + f"; kernel sources {meta['kernel_sources']}")
