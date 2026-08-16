#!/usr/bin/env python3
"""Parse RoboCasa365 criterion-4 v2 results into a 5-config x 4-task SR table.

Reads runs/robocasa365_eval/v2/crit4_<config>_<h1|h2>.jsonl (one JSON per
trial: {task, trial, seed, success, steps}), groups by config and task, and
prints:
  - per-config-per-task success rate (n/N)
  - per-config overall SR (task-macro-averaged)
  - criterion-4 verdict: relative closed-loop comparison CS+CKA vs CS-only
    and CKA-only (diagnostic), incl. binominal CI (Clopper-Pearson, 95%).
"""
import argparse
import glob
import json
import math
import os
import re

CONFIGS = ["fp16", "w6", "csonly", "cscka", "ckaonly"]
CONFIG_LABELS = {
    "fp16": "fp16 (reference)",
    "w6": "uniform W6",
    "csonly": "CS-only",
    "cscka": "CS+CKA(dit)",
    "ckaonly": "CKA-only (diagnostic)",
}
TASK_ORDER = ["OpenCabinet", "OpenStandMixerHead",
              "PickPlaceDrawerToCounter", "CoffeeSetupMug"]


def cp_ci(k, n, z=1.959963984540054):
    """Two-sided 95% Wilson CI (smooth analogue of Clopper-Pearson)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    if k == 0:
        lo = 0.0
    else:
        f = (k * (z ** 2)) / ((n - k + 1) + k * (z ** 2) / n)
        lo = 1.0 / (1.0 + (n - k + 1) / (k * (z ** 2) / n)) if False else (
            1.0 / (1.0 + (z ** 2) / n)) if False else p  # fallthrough
        lo = (2 * k + z ** 2 - z * math.sqrt(z ** 2 + 4 * k * (1 - p))) / (2 * (n + z ** 2))
    if k == n:
        hi = 1.0
    else:
        hi = (2 * k + z ** 2 + z * math.sqrt(z ** 2 + 4 * k * (1 - p))) / (2 * (n + z ** 2))
    return (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/robocasa365_eval/v2")
    ap.add_argument("--repo", default=os.environ.get("REPO", "/home1/gyy/vla/QuantVLA"))
    args = ap.parse_args()
    d = os.path.join(args.repo, args.dir)

    results = {c: {t: {"ok": 0, "n": 0, "steps": []} for t in TASK_ORDER}
               for c in CONFIGS}
    files = sorted(glob.glob(os.path.join(d, "crit4_*.jsonl")))
    dropped = 0
    for f in files:
        base = os.path.basename(f)
        m = re.match(r"crit4_([a-z0-9]+)_h[12]\.jsonl", base)
        if not m:
            continue
        cfg = m.group(1)
        if cfg not in results:
            print(f"!! unknown config in {base}")
            continue
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    dropped += 1
                    continue
                t = r.get("task")
                if t not in results[cfg]:
                    print(f"!! unknown task {t} in {base}")
                    continue
                if r.get("crashed"):
                    # crash-tolerant driver marks env-crash trials; they carry
                    # no policy signal -> excluded from the denominator
                    continue
                results[cfg][t]["n"] += 1
                if r.get("success"):
                    results[cfg][t]["ok"] += 1
                results[cfg][t]["steps"].append(r.get("steps", 0))

    print("=" * 88)
    print("RoboCasa365 criterion-4 v2 — per-task success rate (ok/N, mean steps)")
    print("=" * 88)
    hdr = "config".ljust(26)
    for t in TASK_ORDER:
        hdr += t.replace("PickPlaceDrawerToCounter", "DrawerToCounter").ljust(16)
    hdr += "overall".ljust(14)
    print(hdr)
    totals = {}
    for c in CONFIGS:
        row = CONFIG_LABELS[c].ljust(26)
        oks = 0
        ns = 0
        for t in TASK_ORDER:
            s = results[c][t]
            oks += s["ok"]; ns += s["n"]
            ms = f"{sum(s['steps']) / len(s['steps']):.0f}" if s["steps"] else "-"
            row += f"{s['ok']}/{s['n']} ({ms})".ljust(16)
        # task-macro SR
        macro = [results[c][t]["ok"] / results[c][t]["n"]
                 for t in TASK_ORDER if results[c][t]["n"] > 0]
        macro_sr = sum(macro) / len(macro) if macro else 0.0
        row += f"{macro_sr:.3f}".ljust(14)
        totals[c] = (oks, ns, macro_sr)
        print(row)

    print()
    print("config".ljust(26) + "ok/N      SR     95% CI")
    for c in CONFIGS:
        oks, ns, macro = totals[c]
        lo, hi = cp_ci(oks, ns)
        print(f"{CONFIG_LABELS[c]:26}{oks:>4}/{ns:<5} {oks/ns if ns else 0:.3f}   [{lo:.3f}, {hi:.3f}]")

    print()
    print("criterion-4 verdict (relative closed-loop, task-macro SR):")
    refs = {c: totals[c][2] for c in ["csonly", "cscka", "ckaonly", "w6", "fp16"]}
    print(f"  proxy order (D_func): ckaonly 0.095 < cscka 0.150 < csonly 0.164")
    print(f"  closed-loop:  cscka={refs['cscka']:.3f}  csonly={refs['csonly']:.3f}  "
          f"ckaonly={refs['ckaonly']:.3f}  w6={refs['w6']:.3f}  fp16={refs['fp16']:.3f}")
    if refs["cscka"] > refs["csonly"]:
        verdict = "CS+CKA beats CS-only closed-loop -> criterion 4 PASS"
    else:
        verdict = "CS+CKA does NOT beat CS-only closed-loop -> criterion 4 FAIL " \
                  "(CKA not re-enabled; record D-0xx)"
    print(f"  -> {verdict}")
    if dropped:
        print(f"  (dropped {dropped} malformed lines)")


if __name__ == "__main__":
    main()
