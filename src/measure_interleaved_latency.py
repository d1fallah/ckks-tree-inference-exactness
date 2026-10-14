#!/usr/bin/env python3
"""
Added analysis S2.3a (docs/step2_prespec_m1.md, section C): re-time the original and the coefficient-folding
evaluators in INTERLEAVED order on one day, with the unchanged protocol of measure_ckks_latency.py
(seed-42 hypertension model, 5 discarded warm-up evaluations, 20 timed evaluations on the frozen sample, one
fresh process per degree).

This script only orchestrates: each measurement is a `measure_ckks_latency.py --child K` process, with
M1_LATENCY_EVALUATOR=folded for the folding evaluator. Two rounds in opposite order (original first, then
folding first) so that machine drift is not aligned with one evaluator. The original evaluator does not fit
k = 9 on the locked chain, so k = 9 is folding-only. One child at a time (about 2 GB each).

Output: results/m1_interleaved_latency.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_interleaved_latency.json")
MARK = "@@CHILD_RESULT@@"
ROUND_1 = [("original", 3), ("folding", 3), ("original", 5), ("folding", 5), ("original", 7), ("folding", 7), ("folding", 9)]
ROUND_2 = [("folding", 9), ("folding", 7), ("original", 7), ("folding", 5), ("original", 5), ("folding", 3), ("original", 3)]


def run_child(evaluator: str, k: int) -> dict:
    env = dict(os.environ)
    if evaluator == "folding":
        env["M1_LATENCY_EVALUATOR"] = "folded"
    else:
        env.pop("M1_LATENCY_EVALUATOR", None)
    proc = subprocess.run([sys.executable, "-u", os.path.join(_HERE, "measure_ckks_latency.py"), "--child", str(k)],
                          capture_output=True, text=True, cwd=_HERE, env=env)
    if proc.returncode != 0:
        print(proc.stdout[-1500:], proc.stderr[-1500:])
        raise SystemExit(f"child {evaluator} k={k} failed")
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith(MARK))
    return json.loads(line[len(MARK):])


def main() -> int:
    t_start = time.perf_counter()
    runs = []
    for rnd, order in ((1, ROUND_1), (2, ROUND_2)):
        for evaluator, k in order:
            t0 = time.perf_counter()
            res = run_child(evaluator, k)
            wall = res["timing_per_patient"]["wall_clock_total_s"]
            runs.append({"round": rnd, "order_in_session": len(runs) + 1, "evaluator": evaluator, "k": k,
                         "clock_started_s_since_script_start": t0 - t_start, **res})
            print(f"round {rnd} {evaluator:9s} k={k}: wall_clock {wall['mean_s']:.3f} +/- {wall['sd_s']:.3f} s", flush=True)
    out = {
        "protocol": "docs/step2_prespec_m1.md section C (added analysis S2.3a); per-run protocol of measure_ckks_latency.py",
        "backend": "REAL TenSEAL/SEAL ciphertext execution",
        "order": {"round_1": ROUND_1, "round_2": ROUND_2},
        "runs": runs,
        "total_runtime_seconds": time.perf_counter() - t_start,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
