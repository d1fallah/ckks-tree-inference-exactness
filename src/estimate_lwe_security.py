"""
M1 -- INDEPENDENT LWE security estimate for the locked CKKS parameters (closure guide 4.2).

SEAL's own enforcement (TC128 passes, TC192 fails; results/m1_security_level_verification.json) STANDS and is not replaced. This
adds a second, independent line of evidence: the lattice-estimator of Albrecht, Player and Scott, run on the locked parameters.

Runs in the SEPARATE Sage venv (~/.venvs/m1-estimator, passagemath) against a pinned checkout of malb/lattice-estimator
(~/lattice-estimator). Reads results/m1_ckks_modulus_chain.json (written by dump_ckks_modulus_chain.py in the m1 venv).

Model (stated so the estimate is reproducible and checkable):
  * ring dimension n = N = 16384 (RLWE treated as LWE with n = N; the standard conservative treatment);
  * modulus q = the FULL product of every prime in the chain, including the special prime (400 bits);
  * secret: ternary uniform {-1,0,1} (ND.UniformMod(3)); error: discrete Gaussian, sigma = 3.2 (SEAL documented defaults);
  * attacks: the estimator's primal uSVP, primal BDD, dual (and hybrid variants where tractable), lowest cost reported;
  * cost models: MATZOV (the estimator default, the strongest published) AND ADPS16 classical core-SVP (the model the HE Security
    Standard tables were derived under), so the comparison with the standard is like for like.
  * Reference points from the SAME estimator: the HE Standard's own 128-bit ceiling for N=16384 (438 bits), and the locked chain.

The estimated security level in bits is min over attacks of log2(cost in ring operations). Target: >= 128.

Output: results/m1_lwe_security_estimate.json
"""
import json
import math
import os
import subprocess
import sys
import time

EST_DIR = os.path.expanduser("~/lattice-estimator")
sys.path.insert(0, EST_DIR)

from estimator import LWE, ND, RC  # noqa: E402  (lattice-estimator)

_HERE = os.path.dirname(os.path.abspath(__file__))
CHAIN_PATH = os.path.join(_HERE, "..", "results", "m1_ckks_modulus_chain.json")
OUT_PATH = os.path.join(_HERE, "..", "results", "m1_lwe_security_estimate.json")

TARGET_BITS = 128
HES_CEILING_BITS_N16384 = 438  # HE Security Standard, ternary secret, 128-bit classical: max log2(q) at N=16384
SIGMA = 3.2


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", EST_DIR, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_estimate(name: str, q: int, n: int, cost_model, cost_model_name: str) -> dict:
    """Runs LWE.estimate and returns, per attack, the cost in bits (log2 rop) plus the minimum. q is an exact integer."""
    params = LWE.Parameters(n=n, q=q, Xs=ND.UniformMod(3), Xe=ND.DiscreteGaussian(SIGMA), tag=name)
    t0 = time.time()
    res = LWE.estimate(params, red_cost_model=cost_model, deny_list=("arora-gb", "bkw"))
    per_attack = {}
    for atk, r in res.items():
        rop = float(r["rop"]) if "rop" in r else None
        if rop is not None and rop > 0:
            per_attack[atk] = {"log2_rop": round(math.log2(rop), 3),
                               **{k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in r.items()
                                  if k in ("beta", "d", "zeta", "m", "repeat")}}
    best = min(per_attack.items(), key=lambda kv: kv[1]["log2_rop"])
    return {"parameter_set": name, "n": n, "log2_q": round(math.log2(q), 6), "cost_model": cost_model_name,
            "estimated_security_bits": best[1]["log2_rop"], "cheapest_attack": best[0], "per_attack": per_attack,
            "meets_128_bit_target": bool(best[1]["log2_rop"] >= TARGET_BITS), "estimator_seconds": round(time.time() - t0, 1)}


def main() -> int:
    with open(CHAIN_PATH, "r", encoding="utf-8") as f:
        chain = json.load(f)
    n = chain["N_poly_modulus_degree"]
    q_chain = math.prod(int(p) for p in chain["primes"])  # exact product of every prime, special prime included
    log2_q = math.log2(q_chain)

    runs = []
    for cm, cm_name in ((RC.ADPS16, "ADPS16 classical core-SVP (HE Standard basis)"), (RC.MATZOV, "MATZOV (estimator default)")):
        runs.append(run_estimate("locked chain (400 bits)", q_chain, n, cm, cm_name))
        runs.append(run_estimate("HE Standard 128-bit ceiling at N=16384 (438 bits), reference", 2 ** HES_CEILING_BITS_N16384, n, cm, cm_name))
        print(f"{cm_name}: locked chain -> {runs[-2]['estimated_security_bits']} bits ({runs[-2]['cheapest_attack']}); "
              f"HES ceiling -> {runs[-1]['estimated_security_bits']} bits", flush=True)

    # Toolchain sanity check against a PUBLISHED HE Security Standard row (ternary secret, sigma=3.2): n=1024 admits log2(q)=27 at 128-bit
    # under the classical cost model. Recorded, not asserted, so a large discrepancy is visible rather than hidden.
    sanity = run_estimate("HE Standard published row: n=1024, log2 q=27 (128-bit)", 2 ** 27, 1024, RC.ADPS16, "ADPS16 classical core-SVP (HE Standard basis)")
    print(f"toolchain sanity (n=1024, log2 q=27, ADPS16): {sanity['estimated_security_bits']} bits (published: 128)", flush=True)

    try:
        from importlib.metadata import version as _v
        passagemath = _v("passagemath-standard")
    except Exception:
        passagemath = "unknown"

    out = {
        "purpose": "independent addition to (not replacement of) SEAL's TC128 enforcement; closure guide 4.2",
        "target_security_bits": TARGET_BITS,
        "parameters": {"N": n, "primes": chain["primes"], "prime_bit_lengths": chain["prime_bit_lengths"], "scale_bits": chain["scale_bits"],
                       "log2_q_full_chain": log2_q, "secret": "ternary uniform {-1,0,1}", "error": f"discrete Gaussian sigma={SIGMA}"},
        "estimator": {"name": "lattice-estimator (Albrecht, Player, Scott)", "repository": "https://github.com/malb/lattice-estimator",
                      "commit": git_commit(), "python": sys.version.split()[0], "sage_distribution": f"passagemath-standard {passagemath}"},
        "invocation": ("LWE.estimate(LWE.Parameters(n=N, q=prod(primes), Xs=ND.UniformMod(3), Xe=ND.DiscreteGaussian(3.2)), "
                       "red_cost_model=<RC.ADPS16 | RC.MATZOV>, deny_list=('arora-gb','bkw'))"),
        "modelling_notes": ["RLWE is treated as LWE with n = N (conservative standard treatment).",
                            "q is the full product of the chain including the special prime.",
                            "Secret and error distributions are SEAL's documented defaults, not read back from the binary."],
        "toolchain_sanity_check": sanity,
        "runs": runs,
        "conclusion": {"locked_chain_bits_by_cost_model": {r["cost_model"]: r["estimated_security_bits"] for r in runs if r["parameter_set"].startswith("locked")},
                       "all_locked_chain_runs_meet_target": all(r["meets_128_bit_target"] for r in runs if r["parameter_set"].startswith("locked"))},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"written -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
