# %% [markdown]
# ## Extra reader: public ConvNeXt-S M448 fold 0, rank-mixed into the public result
#
# The public pipeline above is unchanged (it scored 0.943 for us). Below, its `submission.csv`
# is stashed first. Then `mattiaangeli/rsna-knee-cnx-m448-f0-public` - a ConvNeXt-S (DINOv3
# init, 448 px montages) by the author of the 0.943 recipe, which that recipe does not use -
# runs in a separate process, and its per-finding ranks are mixed in with **one global weight**.
# Any failure, coverage gap or time shortfall keeps the public file byte-identical.

# %%
import base64 as _mb64, json as _mjson, os as _mos, shutil as _mshutil, subprocess as _msp, sys as _msys, time as _mtime
from pathlib import Path as _MPath
import pandas as _mpd

_M_TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
              "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
_M_WORK = _MPath("/kaggle/working")
_M_W = float(_mos.environ.get("RSNA_M448_W", "__M448_W__"))
_M_MIN_COVER = 0.98          # share of test studies the reader must predict
_M_NEED_H = 1.0              # hours that must be left before the reader is started


def _m_blend(pub, leg, w):
    """Rank mix. `pub` and `leg` are indexed by study; a study the reader could not predict
    keeps its public rank, so it is neither helped nor hurt."""
    pr = pub[_M_TARGETS].rank(pct=True)
    lr = leg[_M_TARGETS].reindex(pub.index).rank(pct=True)
    lr = lr.where(lr.notna(), pr)
    return ((1 - w) * pr + w * lr).rank(pct=True)


def _m_gate(leg, n, rec):
    """Reasons not to use the reader's output (empty list = use it)."""
    why = []
    if rec.get("timed_out"):
        why.append("reader ran out of time")
    covered = int(leg[_M_TARGETS].notna().all(axis=1).sum())
    if covered < _M_MIN_COVER * n:
        why.append(f"reader predicted {covered}/{n} studies")
    if n > 10 and (leg[_M_TARGETS].nunique() < 0.5 * covered).any():
        why.append("reader has a near-constant column")
    return why


def _m_age_h():
    up = float(open("/proc/uptime").read().split()[0])
    start = int(open("/proc/self/stat").read().split(")")[-1].split()[19]) / _mos.sysconf("SC_CLK_TCK")
    return (up - start) / 3600


def _m_find_root():
    for d, dirs, files in _mos.walk("/kaggle/input"):
        dirs[:] = [x for x in dirs if x not in ("train_series", "test_series")]
        if "btkr_cnx_montage448_manifest.json" in files:
            return d
    raise FileNotFoundError("mattiaangeli/rsna-knee-cnx-m448-f0-public is not attached")


def _m_main():
    sub_path, stash = _M_WORK / "submission.csv", _M_WORK / "_public_submission.csv"
    pub = _mpd.read_csv(sub_path, dtype={"StudyInstanceUID": str})
    assert pub.columns.tolist() == ["StudyInstanceUID", *_M_TARGETS], "public schema drift"
    assert pub[_M_TARGETS].notna().all().all(), "public submission has nulls"
    _mshutil.copy2(sub_path, stash)
    print(f"[m448] stashed public submission {pub.shape}; weight {_M_W}", flush=True)
    ok, why = False, "not attempted"
    try:
        left = 9.0 - _m_age_h() - 0.4
        if left < _M_NEED_H:
            raise RuntimeError(f"only {left:.2f} h left after the public pipeline")
        script = _M_WORK / "_m448_leg.py"
        script.write_bytes(_mb64.b64decode("__LEG_B64__"))
        out = _M_WORK / "_m448_leg.csv"
        budget = max(600.0, left * 3600 - 600)        # the reader stops itself before the hard timeout
        t0 = _mtime.time()
        rc = _msp.run([_msys.executable, "-u", str(script)], timeout=int(left * 3600),
                      env={**_mos.environ, "M448_ROOT": _m_find_root(), "M448_OUT": str(out),
                           "M448_BUDGET_S": str(int(budget))}).returncode
        print(f"[m448] reader rc={rc} in {(_mtime.time() - t0) / 60:.1f} min", flush=True)
        if rc != 0:
            raise RuntimeError(f"reader exited {rc}")
        rec = _mjson.loads(out.with_suffix(".json").read_text())
        leg = _mpd.read_csv(out, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
        leg = leg.reindex(pub["StudyInstanceUID"])        # coverage is counted over the public rows
        bad = _m_gate(leg, len(pub), rec)
        if bad:
            raise RuntimeError("; ".join(bad))
        mixed = _m_blend(pub.set_index("StudyInstanceUID"), leg, _M_W)
        final = mixed.reset_index()[["StudyInstanceUID", *_M_TARGETS]]
        assert final["StudyInstanceUID"].tolist() == pub["StudyInstanceUID"].tolist()
        assert final[_M_TARGETS].notna().all().all() and len(final) == len(pub)
        final.to_csv(sub_path, index=False)
        ok, why = True, f"blended at weight {_M_W}; reader receipt {rec}"
    except Exception as exc:                           # noqa: BLE001 - any failure keeps the public file
        why = f"{type(exc).__name__}: {exc}"
        _mshutil.copy2(stash, sub_path)
    _mjson.dump({"m448_applied": ok, "why": why}, open(_M_WORK / "_m448_receipt.json", "w"))
    print(f"[m448] applied={ok}: {why}", flush=True)
    check = _mpd.read_csv(sub_path)
    assert check.columns.tolist() == ["StudyInstanceUID", *_M_TARGETS] and check[_M_TARGETS].notna().all().all()
    print("final submission.csv", check.shape)


if _M_WORK.is_dir():
    _m_main()
