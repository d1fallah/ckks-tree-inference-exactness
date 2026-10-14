#!/usr/bin/env python3
r"""
Design constants for the security section (PROPOSED_jisa_security.md): the size of a single-bit probe set of the thermometer protocol, which is the number of
inputs times (B - 1) for each bin count B. It reads only stored aggregate metadata (the list of input names of the continuous-feature task and the bin counts of
the thermometer summary), computes nothing from patient data, and runs no attack. The count is the size of the probe set, not an estimate of the cost of recovering the model.

Output: results/m1c_security_design_constants.json
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def main():
    ct = json.load(open(os.path.join(RES, "m1c_final_summary.json"), encoding="utf-8"))
    tm = json.load(open(os.path.join(RES, "m1c_thermo_final_summary.json"), encoding="utf-8"))
    n_inputs = len(ct["inputs"])
    out = {
        "note": "Size of a single-bit probe set of the thermometer protocol: n_inputs x (B - 1). A design constant. It is not an estimate of the cost of recovering the model, and no attack was run.",
        "n_inputs": n_inputs,
        "single_bit_probe_set_size": {},
    }
    for key in tm["by_B"]:
        B = int(key.split("=")[1])
        out["single_bit_probe_set_size"][key] = n_inputs * (B - 1)
    path = os.path.join(RES, "m1c_security_design_constants.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["single_bit_probe_set_size"]))


if __name__ == "__main__":
    main()
