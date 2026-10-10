#!/usr/bin/env python3
"""Live view of our Kaggle sessions for a terminal: GPU quota, the latest competition submissions
(pending or scored) and the status and latest progress lines of each named job, refreshed every
minute. Read-only: it only asks Kaggle for status, quota, logs and the submission list. Stop it
with Ctrl-C; nothing on Kaggle is affected.

Usage: python3 kaggle/watch_jobs.py rsna-knee-gpjw-train-r6 rsna-knee-gpjw-train-r7
"""
import csv, io, json, os, re, subprocess, sys, time

KAGGLE = os.path.expanduser("~/.local/bin/kaggle")
USER = "natchapon600176"


def sh(args, timeout):
    """stdout of a command; if it is still running at the timeout, what it printed so far."""
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        return out.decode(errors="replace") if isinstance(out, bytes) else out


def log_lines(kernel):
    raw = sh([KAGGLE, "kernels", "logs", "-f", f"{USER}/{kernel}"], 25)
    parts = []
    for m in re.finditer(r'"data":\s*"((?:[^"\\]|\\.)*)"', raw):
        try:
            parts.append(json.loads('"' + m.group(1) + '"'))
        except ValueError:
            pass
    return [l for l in ("".join(parts) if parts else raw).splitlines() if l.strip()]


def submissions(n=4):
    rows = list(csv.DictReader(io.StringIO(sh([KAGGLE, "competitions", "submissions", "rsna-knee-abnormality-detection", "-v"], 30))))
    return [f"   {r['date'][:16]} UTC  ref {r['ref']}  {r['status'].replace('SubmissionStatus.', ''):9s} "
            f"score {r['publicScore'] or '-':6s} {r['description'][:70]}" for r in rows[:n] if r.get("ref")]


while True:
    screen = [time.strftime("%Y-%m-%d %H:%M:%S"), sh([KAGGLE, "quota"], 30).strip(), "\n== latest submissions"] + submissions()
    for kernel in sys.argv[1:]:
        out = sh([KAGGLE, "kernels", "status", f"{USER}/{kernel}"], 30).strip()
        status = out.split("has status")[-1].strip() if "has status" in out else "not started yet"
        screen.append(f"\n== {kernel}: {status}")
        if "RUNNING" in status:
            seen = []
            for line in log_lines(kernel):
                if re.search(r"\| |port check|indexed|ALL JOBS|Traceback|Error", line):
                    key = re.sub(r"^\d\d:\d\d ", "", line)
                    if key not in seen:
                        seen.append(key)
            screen += ["   " + s[:220] for s in seen[-6:]] or ["   (no progress lines yet)"]
    os.system("clear")
    print("\n".join(screen), flush=True)
    time.sleep(60)
