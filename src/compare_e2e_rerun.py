"""Compare an end-to-end re-run (results/) against the committed results (results_committed/). Run inside ~/m1_e2e."""
import hashlib
import json
import os
import re
import statistics
import sys

NEW, OLD = "results", "results_committed"
WIN = sys.argv[1]  # path of the Windows-tree module dir as seen from WSL

# leaf paths whose values legitimately differ between runs: wall-clock, memory, host, provenance
TIMING = re.compile(r"(latenc|second|_s\b|_s_|runtime|elapsed|wall|time|rss|_kb|memory|peak|hardware|cpu|kernel|platform|python|tenseal|"
                    r"env_|per_patient_s|mean_s|sd_s|std_s|throughput)", re.I)
PROVENANCE = re.compile(r"(git|sha256|sha-256|head|hash|dirty|clean|timestamp|date|version|commit)", re.I)


def norm_bytes(path):
    with open(path, "rb") as f:
        return f.read().replace(b"\r\n", b"\n")


def leaves(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, f"{path}[{i}]")
    else:
        yield path, o


def classify(path):
    if PROVENANCE.search(path):
        return "provenance"
    if TIMING.search(path):
        return "timing"
    return "substantive"


def compare_json(a, b):
    la, lb = dict(leaves(a)), dict(leaves(b))
    out = {"timing": [], "provenance": [], "substantive": [], "only_old": [], "only_new": []}
    for p in la.keys() - lb.keys():
        out["only_old"].append(p)
    for p in lb.keys() - la.keys():
        out["only_new"].append(p)
    for p in la.keys() & lb.keys():
        x, y = la[p], lb[p]
        if x == y:
            continue
        cls = classify(p)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            d = abs(x - y)
            rel = d / abs(x) if x else float("inf")
            out[cls].append((p, x, y, d, rel))
        else:
            out[cls].append((p, x, y, None, None))
    return out


report, summary = {}, {"identical": [], "only_timing_or_provenance_differs": [], "substantive_differs": [], "structure_differs": []}
old_files = {f for f in os.listdir(OLD) if f.endswith(".json")}
new_files = {f for f in os.listdir(NEW) if f.endswith(".json")}
for f in sorted(old_files | new_files):
    if f not in new_files:
        summary["structure_differs"].append(f"{f}: in committed, not produced by the run")
        continue
    if f not in old_files:
        summary["structure_differs"].append(f"{f}: produced by the run, not committed")
        continue
    if norm_bytes(f"{NEW}/{f}") == norm_bytes(f"{OLD}/{f}"):
        summary["identical"].append(f)
        continue
    with open(f"{NEW}/{f}", encoding="utf-8") as fa, open(f"{OLD}/{f}", encoding="utf-8") as fb:
        new, old = json.load(fa), json.load(fb)
    c = compare_json(old, new)
    report[f] = c
    if c["only_old"] or c["only_new"]:
        summary["structure_differs"].append(f"{f}: {len(c['only_old'])} keys only in committed, {len(c['only_new'])} only in the run")
    if c["substantive"]:
        summary["substantive_differs"].append(f)
    elif not (c["only_old"] or c["only_new"]):
        summary["only_timing_or_provenance_differs"].append(f)

print(f"JSON files: {len(old_files | new_files)}  byte-identical: {len(summary['identical'])}")
print("identical:", ", ".join(summary["identical"]))
print("\nonly timing/provenance leaves differ:")
for f in summary["only_timing_or_provenance_differs"]:
    t = report[f]["timing"]
    rels = [r for (_, _, _, d, r) in t if r is not None and r != float("inf")]
    print(f"  {f}: {len(t)} timing leaves, {len(report[f]['provenance'])} provenance leaves; "
          f"timing rel. diff median {statistics.median(rels) * 100:.1f}% max {max(rels) * 100:.1f}%" if rels else f"  {f}: {len(t)} timing, {len(report[f]['provenance'])} provenance (non-numeric)")
print("\nSUBSTANTIVE differences:")
for f in summary["substantive_differs"]:
    s = report[f]["substantive"]
    nums = [d for (_, _, _, d, _) in s if d is not None]
    print(f"  {f}: {len(s)} leaves; max abs diff {max(nums) if nums else 'n/a (non-numeric)'}")
    for (p, x, y, d, r) in sorted(s, key=lambda t: -(t[3] or 0))[:6]:
        print(f"      {p}: committed={x!r} run={y!r} diff={d}")
print("\nSTRUCTURE differences:", *summary["structure_differs"], sep="\n  ")

# non-JSON artifacts: models, coefficients, tables vs the committed (Windows) tree
def sha(p):
    return hashlib.sha256(norm_bytes(p)).hexdigest()

extra = {}
mdir = "data/models"
models = sorted(os.listdir(mdir))
diff_models = [m for m in models if not os.path.exists(f"{WIN}/{mdir}/{m}") or sha(f"{mdir}/{m}") != sha(f"{WIN}/{mdir}/{m}")]
extra["models"] = {"n": len(models), "byte_identical_to_committed": len(models) - len(diff_models), "differing": diff_models}
extra["poly_coeffs_v1.json"] = sha("data/poly_coeffs_v1.json") == sha(f"{OLD}/poly_coeffs_v1.json.committed")
tab = sorted(os.listdir("paper_assets/tables"))
extra["tables"] = {t: sha(f"paper_assets/tables/{t}") == sha(f"{WIN}/paper_assets/tables/{t}") for t in tab}
print("\nmodels:", extra["models"]["byte_identical_to_committed"], "/", extra["models"]["n"], "byte-identical to the committed models", extra["models"]["differing"][:5])
print("poly_coeffs_v1.json identical to committed:", extra["poly_coeffs_v1.json"])
print("tables identical to committed (tables regenerated from the RUN's results):", extra["tables"])

with open("results_e2e_comparison.json", "w", encoding="utf-8") as f:
    json.dump({"summary": summary, "extra": extra,
               "differences": {k: {kk: [list(t) if isinstance(t, tuple) else t for t in vv][:200] for kk, vv in v.items()} for k, v in report.items()}},
              f, indent=1, default=str)
print("\nwritten results_e2e_comparison.json")
