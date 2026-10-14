#!/usr/bin/env python3
"""
Generate booktabs LaTeX tables from the real results JSONs, per CLAUDE.md Sec 6/1.3 and the
Q1 Paper Finalization Guide Sec 24 (nine required tables). Every cell goes through
src/paper_data.py's Evidence accessor -- see that module's docstring for why. Nothing here
computes a new number outside of Evidence.reduce()'s declared, replayable aggregations.

T1-T8 are the guide's numbered tables. T9 (the literature comparison, guide Sec 26) is built by
table_t9_related_work() below from paper_assets/related_work.json + M1_Literature_Map.md (19
sources, each verified per CLAUDE.md rule 10 -- see paper/refs/m1_bib_verification.json). Every
external-paper cell goes through Evidence.external(), which requires a (citation, locus) pair and
is independently checked by src/verify_paper_assets.py:check_external_citations(). "This work"'s
own row uses ev.get()/ev.reduce() against real artifacts exactly like T1-T8.
S1-S4 are supplementary tables the guide does not number but that remain useful references.

Output: paper_assets/tables/*.tex + paper_assets/provenance/*.json (one sidecar per table)
Usage: python generate_latex_tables.py
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_data import Evidence, real_latency_by_degree  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "paper_assets", "tables")
RELATED_WORK_JSON = os.path.join(_HERE, "..", "paper_assets", "related_work.json")

# Caption fragments every generator concatenates into its `note=` so check_qc_labels (verify_paper_assets.py)
# can find the required wording without re-parsing LaTeX.
SIMULATED = "simulated"
PROXY = "Proxy cohort (MIMIC-IV-ED-derived)"
K9_NOTE = (r"the $\Delta$F1 of $k=9$ is simulation only; the original evaluator cannot execute $k=9$ (8 levels needed, 7 available), "
           r"and real execution at $k=9$ uses the coefficient-folding evaluator (Table~\ref{tab:t5_real_vs_simulated})")
FOLDED_ARTIFACT = "m1_encrypted_multiseed_folded.json"
# Full test split on real ciphertexts (revision E2). The override exists only to test the generator on the monitoring file.
E2_ARTIFACT = os.environ.get("M1_E2_ARTIFACT", "m1_real_full_split.json")
K9_NOTE_SHORT = (r"The $\Delta$F1 of $k=9$ is simulation only, and the original evaluator cannot execute $k=9$ "
                 r"(8 levels needed, 7 available).")
K1_NOTE = "$k=1$ is an additional diagnostic arm outside the evaluated degree family $\\{3,5,7,9\\}$"


def bf(cell: str) -> str:
    """Bold a formatted cell (supervisor rule: the best result in a performance table is bold)."""
    return "\\textbf{" + cell + "}"


def best_idx(values, higher: bool = True) -> set:
    """Indices of the best value(s) among the non-None entries; ties are all returned. When every entry ties,
    nothing is best, so nothing is bold."""
    vals = [(i, v) for i, v in enumerate(values) if v is not None]
    if not vals:
        return set()
    target = max(v for _, v in vals) if higher else min(v for _, v in vals)
    winners = {i for i, v in vals if abs(v - target) < 1e-12}
    return set() if len(winners) == len(vals) else winners


# Plain-language names for caption text (a code identifier never appears in a caption).
LABEL_DISPLAY = {"label_hypertension_uncomplicated": "uncomplicated hypertension"}


def tex_escape(value: str) -> str:
    """Escape a raw JSON string value (a label name, a filename, an artifact path) for use in
    LaTeX TEXT mode -- an unescaped '_' outside math mode is a "Missing $ inserted" compile
    error (found empirically: label_hypertension_uncomplicated inside \\texttt{} and
    results/m1_depth_probe.json in plain caption text both broke the stub compile). Applied at
    every call site that splices a bare ev.get() string result into a table's visible text;
    numeric fmt=... results never need this since they contain only digits/./,/-."""
    return (value.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("#", r"\#")
            .replace("%", r"\%").replace("&", r"\&"))


def write(name: str, content: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"wrote {path}")


def finish(asset_id: str, filename: str, lines: list[str], ev: Evidence, *, wide: bool = True) -> None:
    """`wide=True` (the default) renders the float as `table*` (full page width) instead of the
    plain `table` (one IEEEtran column, ~3.5in). Established empirically: even T1, a 2-column
    table, overflowed a single column by 137pt on a stub-compile test (one cell's text -- "134
    (ATC-1: 14, ATC-2: 86, ...)" -- is simply too long for 3.5in), so every generated table here
    defaults to full width rather than guessing per-table which ones are narrow enough. This is a
    single string substitution at the ONE place all tables are finished, not a per-table hand-edit
    -- the alternative of a `\\let\\table\\@dblfloat` trick at the .tex \\input call site was tried
    and rejected: `\\@dblfloat` is not a drop-in substitute for `\\table` (fails with "Missing
    number, treated as zero" from `\\ftype@`), so the environment must be chosen at generation time."""
    content = "\n".join(lines) + "\n"
    if wide:
        assert content.count("\\begin{table}[t]") == 1 and content.count("\\end{table}") == 1, asset_id
        content = content.replace("\\begin{table}[t]", "\\begin{table*}[t]", 1)
        content = content.replace("\\end{table}", "\\end{table*}", 1)
    write(filename, content)
    sidecar_path = ev.write_provenance()
    print(f"  provenance -> {sidecar_path}")


# ---------------------------------------------------------------------------------------------
# T1 -- Dataset and clinical task
# ---------------------------------------------------------------------------------------------
def table_t1_dataset_task() -> None:
    ev = Evidence("t1_dataset_task")
    n_stays = ev.get("m1_cohort_flow.json", "/final_cohort_size", fmt=",", note=f"{PROXY}. {SIMULATED} n/a (dataset description).")
    n_feat = ev.get("m1_plaintext_baseline.json", "/feature_dim", note="total feature dimension")
    n_atc1 = ev.get("m1_feature_space.json", "/n_atc1_flags", note="ATC level-1 flags")
    n_atc2 = ev.get("m1_feature_space.json", "/n_atc2_flags", note="ATC level-2 flags")
    n_labels = ev.get("m1_label_space.json", "/top_k", note="Elixhauser labels selected by prevalence")
    seeds = ev.get("m1_plaintext_baseline.json", "/seeds", note="model seeds")
    n_train = ev.get("m1_plaintext_baseline.json", "/split/train", fmt=",", note="train stays")
    n_val = ev.get("m1_plaintext_baseline.json", "/split/val", fmt=",", note="validation stays")
    n_test = ev.get("m1_plaintext_baseline.json", "/split/test", fmt=",", note="test stays")
    label_rows = []
    for i in range(5):
        cat = ev.get("m1_label_space.json", f"/full_prevalence_table_all_elixhauser_categories/{i}/category")
        prev = ev.get("m1_label_space.json", f"/full_prevalence_table_all_elixhauser_categories/{i}/prevalence", fmt=".4f")
        label_rows.append((cat, prev))
    coverage = ev.get("m1_atc_crosswalk_coverage.json", "/row_weighted_coverage_rate", fmt=".4f",
                       note="row-weighted NDC->ATC medication-code coverage")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Dataset and prediction task. " + PROXY + r"; a research cohort, not a "
        r"clinical-validation dataset. Patient-level split; five random seeds.}",
        r"\label{tab:t1_dataset_task}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"ED stays in cohort & {n_stays} \\\\",
        f"Feature dimension & {n_feat} (ATC-1: {n_atc1}, ATC-2: {n_atc2}, + ETC/aggregate/demographic) \\\\",
        f"Medication-code coverage (row-weighted) & {coverage} \\\\",
        f"Prediction labels (Elixhauser, top-{n_labels} by prevalence) & see below \\\\",
        f"Train / validation / test stays (60/20/20 by patient) & {n_train} / {n_val} / {n_test} \\\\",
        f"Model seeds & {{{ev.literal(', '.join(str(s) for s in seeds), reason='joined display of the seeds already recorded above (splits.MODEL_SEEDS)')}}} \\\\",
        r"\midrule",
        r"\multicolumn{2}{l}{\textit{Label prevalence (fraction of stays)}} \\",
    ]
    for cat, prev in label_rows:
        lines.append(f"\\quad {cat} & {prev} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t1_dataset_task", "t1_dataset_task.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T2 -- Plaintext baseline (ported from the pre-existing table_plaintext_baseline; fixes the
# defect where per-label means were recomputed from per_seed_results instead of reading the
# artifact's own per_label_across_seeds aggregate -- two derivations of one number, now one).
# ---------------------------------------------------------------------------------------------
def table_t2_plaintext_baseline() -> None:
    ev = Evidence("t2_plaintext_baseline")
    label_cols = ev.get("m1_plaintext_baseline.json", "/label_cols")
    n_trees = ev.get("m1_plaintext_baseline.json", "/n_estimators")
    max_depth = ev.get("m1_plaintext_baseline.json", "/max_depth")
    macro_mean = ev.get("m1_plaintext_baseline.json", "/macro_f1_tuned/mean", fmt=".4f")
    macro_std = ev.get("m1_plaintext_baseline.json", "/macro_f1_tuned/std", fmt=".4f")
    macro05_mean = ev.get("m1_plaintext_baseline.json", "/macro_f1_at_0.5/mean", fmt=".4f")
    macro05_std = ev.get("m1_plaintext_baseline.json", "/macro_f1_at_0.5/std", fmt=".4f")
    auroc_mean = ev.get("m1_plaintext_baseline.json", "/mean_auroc/mean", fmt=".4f")
    auroc_std = ev.get("m1_plaintext_baseline.json", "/mean_auroc/std", fmt=".4f",
                        note="registered for README cross-reference; not shown in this table's compact columns")
    auprc_mean = ev.get("m1_plaintext_baseline.json", "/mean_auprc/mean", fmt=".4f")
    auprc_std = ev.get("m1_plaintext_baseline.json", "/mean_auprc/std", fmt=".4f")
    threshold_note = ev.literal("0.5", reason="the fixed decision threshold this row's metric uses "
                                "(as opposed to the per-label tuned threshold in the row above); "
                                "a declared evaluation setting, not a measured value")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Plaintext baseline, {{{PROXY}}}, "
        rf"5 seeds, {n_trees} trees / depth {max_depth}, one binary XGBoost model per label. "
        r"Plaintext inference without encryption.}",
        r"\label{tab:t2_plaintext_baseline}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Label & Test prevalence & F1 (tuned, mean $\pm$ SD) & AUROC & AUPRC \\",
        r"\midrule",
    ]
    for lab_key in label_cols:
        lab_name = lab_key.replace("label_", "").replace("_", " ")
        base = f"/per_label_across_seeds/{lab_key}"
        prev = ev.get("m1_plaintext_baseline.json", f"{base}/test_prevalence", fmt=".3f")
        f1_mean = ev.get("m1_plaintext_baseline.json", f"{base}/f1_tuned/mean", fmt=".4f")
        f1_std = ev.get("m1_plaintext_baseline.json", f"{base}/f1_tuned/std", fmt=".4f")
        auroc = ev.get("m1_plaintext_baseline.json", f"{base}/auroc/mean", fmt=".4f")
        auprc = ev.get("m1_plaintext_baseline.json", f"{base}/auprc/mean", fmt=".4f")
        lines.append(f"{lab_name} & {prev} & {f1_mean} $\\pm$ {f1_std} & {auroc} & {auprc} \\\\")
    lines += [
        r"\midrule",
        f"\\textbf{{Macro (mean over 5 seeds)}} & -- & \\textbf{{{macro_mean} $\\pm$ {macro_std}}} & {auroc_mean} & {auprc_mean} \\\\",
        f"Macro F1 at fixed 0.5 threshold (untuned) & -- & {macro05_mean} $\\pm$ {macro05_std} & -- & -- \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    finish("t2_plaintext_baseline", "t2_plaintext_baseline.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T3 -- CKKS configuration and circuit parameters
# ---------------------------------------------------------------------------------------------
def table_t3_ckks_circuit_config() -> None:
    ev = Evidence("t3_ckks_circuit_config")
    ring_degree = ev.get("m1_ckks_modulus_chain.json", "/N_poly_modulus_degree", fmt=",")
    primes = ev.get("m1_ckks_modulus_chain.json", "/prime_bit_lengths")
    total_bits = ev.get("m1_ckks_modulus_chain.json", "/total_modulus_bits_sum_of_bit_lengths")
    scale_bits = ev.get("m1_ckks_modulus_chain.json", "/scale_bits")
    tenseal_ver = ev.get("m1_ckks_modulus_chain.json", "/tenseal_version")
    a_half = ev.get("m1_circuit_params.json", "/approximation_interval/a_half_width", fmt=".6f")
    interval_pctl = ev.get("m1_circuit_params.json", "/approximation_interval/interval_percentile_preregistered")
    norm_rule = ev.get("m1_circuit_params.json", "/normalization_rule")
    max_leaves = ev.reduce("max", "m1_circuit_params.json", "/split_census/per_model/*/n_leaves", fmt=",")
    slots = ev.get("m1_circuit_params.json", "/depth_budget/k=5/slots_at_that_degree", fmt=",")
    tree_depth = ev.get("m1_circuit_params.json", "/tree_depth")
    tc128_max = ev.get("m1_security_level_verification.json", "/by_security_level/TC128/standard_max_bits_for_N")
    tc128_set = ev.get("m1_security_level_verification.json", "/by_security_level/TC128/parameters_set_under_enforcement")
    tc192_set = ev.get("m1_security_level_verification.json", "/by_security_level/TC192/parameters_set_under_enforcement")
    gamma = ev.get("m1_encrypted_multiseed.json", "/coefficients/gamma", fmt=".6f")
    anchor_pctl = ev.get("m1_encrypted_multiseed.json", "/coefficients/gamma_anchor_percentile")
    coeff_ver = ev.get("m1_encrypted_multiseed.json", "/coefficients/version")
    sec_level = ev.literal(128, reason="notation.LAM_SECURITY_LEVEL, the locked target security level "
                            "(distinct from the TC128/TC192 Homomorphic Encryption Security Standard names, "
                            "which are themselves literal standard names, not measurements)")
    tc192_name = ev.literal(192, reason="TC192 is a standard name (Homomorphic Encryption Security Standard), "
                             "not a measured value")

    occupancy = ev.literal(f"{int(max_leaves.replace(',', '')) / int(slots.replace(',', '')) * 100:.1f}",
                            reason="derived ratio of two already-recorded values (max_leaves / slots), "
                            "not a new source read; both operands are recorded above")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{CKKS configuration and circuit parameters (SEAL/TenSEAL {tenseal_ver}), fixed for every "
        rf"experiment. {PROXY}. Utility values are {SIMULATED}; the chain, slot and security parameters "
        r"are those of the executed configuration. "
        r"Packing: leaf-aligned, one slot per leaf, one ciphertext per tree level, zero rotations in the "
        r"packing and the path product (the final ensemble sum is one rotate-and-add reduction).}",
        r"\label{tab:t3_ckks_circuit_config}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Parameter & Value \\",
        r"\midrule",
        f"Ring degree $N$ & {ring_degree} \\\\",
        f"Coefficient modulus chain (bits) & [{', '.join(str(b) for b in primes)}] (sum {total_bits}) \\\\",
        f"Scale & $2^{{{scale_bits}}}$ \\\\",
        f"Security level & {sec_level}-bit (TC{sec_level} SEAL-enforced: {tc128_set}, ceiling {tc128_max} bits; "
        f"TC{tc192_name} satisfied: {tc192_set}) \\\\",
        f"Approximation interval $[-a,a]$ & $a={a_half}$ (pre-specified percentile {interval_pctl} of "
        f"train-split margin quantiles) \\\\",
        f"Comparison polynomial coefficients & $\\gamma={gamma}$ (anchor percentile "
        f"{anchor_pctl}, selected on the validation split) \\\\",
        f"Normalization & {norm_rule} \\\\",
        f"Tree depth (levels per model) & {tree_depth} \\\\",
        f"Leaves per model (max over 25 frozen models) & {max_leaves} of {slots} slots ({occupancy}\\% occupancy) \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    finish("t3_ckks_circuit_config", "t3_ckks_circuit_config.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T4 -- Degree x {utility, latency, depth, feasibility}, k=1 included as a labelled extra arm
# ---------------------------------------------------------------------------------------------
def table_t4_degree_grid() -> None:
    ev = Evidence("t4_degree_grid")
    k1_note_registered = ev.literal(K1_NOTE, reason="fixed descriptive text naming the locked degree "
                                    "family {3,5,7,9} (notation.K_APPROX_DEGREE_VALUES); not a measured value")
    k9_note_registered = ev.literal(K9_NOTE, reason="fixed descriptive text; depth counts are separately "
                                    "recorded above via m1_depth_probe.json")
    tolerance = ev.literal("0.03", reason="notation.MAX_ACCEPTABLE_F1_LOSS, the locked utility tolerance")
    lat_orig = real_latency_by_degree(ev)
    lat_fold = real_latency_by_degree(ev, FOLDED_ARTIFACT)
    available = ev.get("m1_depth_probe.json", "/per_degree/k=3/levels_available_on_locked_chain",
                        note="levels available on the locked chain")
    n_real = ev.get("m1_encrypted_multiseed.json", "/n_per_seed", note="real-ciphertext patients per seed")
    yes, no = r"\checkmark", r"$\times$"

    # each row: [k, dF1, CI, lat_orig, depth_orig, fits_orig, lat_fold, depth_fold, fits_fold]; raw values drive the bolding
    rows, raw = [], []
    d1_mean = ev.get("m1_ckks_sweep_k1_simulated.json", "/delta_f1_mean_over_seeds", fmt=".4f", note=f"{SIMULATED}. {K1_NOTE}.")
    d1_lo = ev.get("m1_ckks_sweep_k1_simulated.json", "/paired_bootstrap/ci_95_lower", fmt=".4f")
    d1_hi = ev.get("m1_ckks_sweep_k1_simulated.json", "/paired_bootstrap/ci_95_upper", fmt=".4f")
    d1_depth = ev.get("m1_depth_budget_k1.json", "/code_derived/total_multiplicative_depth")
    d1_fits = ev.get("m1_depth_budget_k1.json", "/fits_locked_chain")
    rows.append(["1$^\\dagger$", d1_mean, f"[{d1_lo}, {d1_hi}]", "n/a", str(d1_depth), yes if d1_fits else no, "n/a", "n/a", "n/a"])
    raw.append(None)  # the diagnostic arm is outside the evaluated family and is not a candidate for the best value
    for degree in (3, 5, 7, 9):
        key = f"k={degree}"
        mean = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/delta_f1_mean_over_seeds", fmt=".4f",
                       note=SIMULATED if degree != 9 else f"{SIMULATED}. {K9_NOTE}.")
        lo = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/paired_bootstrap/ci_95_lower", fmt=".4f")
        hi = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/paired_bootstrap/ci_95_upper", fmt=".4f")
        depth_o = ev.get("m1_depth_probe.json", f"/per_degree/{key}/code_derived/total_multiplicative_depth")
        fits_o = ev.get("m1_depth_probe.json", f"/per_degree/{key}/fits_locked_chain")
        depth_f = ev.get("m1_folded_depth_accounting.json", f"/per_degree/{key}/total_depth_folded")
        fits_f = ev.get("m1_folded_depth_accounting.json", f"/per_degree/{key}/fits_folded")
        lo_o = f"{lat_orig[degree][0]:.3f} $\\pm$ {lat_orig[degree][1]:.3f}" if degree in lat_orig else "n/a$^\\ddagger$"
        lo_f = f"{lat_fold[degree][0]:.3f} $\\pm$ {lat_fold[degree][1]:.3f}"
        rows.append([str(degree), mean, f"[{lo}, {hi}]", lo_o, str(depth_o), yes if fits_o else no,
                     lo_f, str(depth_f), yes if fits_f else no])
        raw.append({"df1": float(mean), "lat_o": lat_orig[degree][0] if degree in lat_orig else None,
                    "depth_o": depth_o, "lat_f": lat_fold[degree][0], "depth_f": depth_f})
    for col, rawkey in ((1, "df1"), (3, "lat_o"), (4, "depth_o"), (6, "lat_f"), (7, "depth_f")):
        for i in best_idx([r[rawkey] if r else None for r in raw], higher=False):
            rows[i][col] = bf(rows[i][col])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Polynomial degree $k$ versus predictive utility, latency on real ciphertexts, "
        rf"multiplicative depth, and feasibility on the fixed modulus chain ({available} levels available), for the "
        rf"original and the coefficient-folding polynomial evaluator. "
        f"$\\Delta$F1 is {SIMULATED} (full test split, 5 seeds; tolerance $\\le 0.03$) and does not depend on the evaluator. "
        rf"Latency is the mean $\pm$ SD across 5 seeds $\times$ {n_real} patients for one label, measured on real ciphertexts. "
        rf"Lowest value per column in bold, among $k\in\{{3,5,7,9\}}$. "
        rf"{PROXY}. $\dagger$ {K1_NOTE}. $\ddagger$ {K9_NOTE}.}}",
        r"\label{tab:t4_degree_grid}",
        r"\begin{tabular}{lrlrrcrrc}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Original evaluator} & \multicolumn{3}{c}{Folding evaluator} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
        r"$k$ & $\Delta$F1 & 95\% CI & Real latency (s) & Depth & Fits chain & Real latency (s) & Depth & Fits chain \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t4_degree_grid", "t4_degree_grid.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T5 -- Real-versus-simulated CKKS validation
# ---------------------------------------------------------------------------------------------
def table_t5_real_vs_simulated() -> None:
    ev = Evidence("t5_real_vs_simulated")
    k9_note_registered = ev.literal(K9_NOTE, reason="fixed descriptive text; depth counts are separately "
                                    "recorded via m1_depth_probe.json in Table 4")
    n_seeds = ev.get("m1_encrypted_multiseed.json", "/seeds")
    n_per_seed = ev.get("m1_encrypted_multiseed.json", "/n_per_seed")
    label = ev.get("m1_encrypted_multiseed.json", "/label_same_for_all_seeds")

    arms = (("Original", "m1_encrypted_multiseed.json"), ("Folding", FOLDED_ARTIFACT))
    rows, raw = [], []
    n_total = 0
    for arm, artifact in arms:
        degrees = ev.get(artifact, "/k_executed")
        for degree in degrees:
            key = f"k={degree}"
            base = f"/pooled_across_seeds/{key}"
            agree = ev.get(artifact, f"{base}/agreement_real_vs_simulated/mean", fmt=".4f")
            mae = ev.get(artifact, f"{base}/normalized_mae_real_vs_simulated/mean", fmt=".2e")
            maxerr = ev.get(artifact, f"{base}/max_abs_error_real_vs_simulated/mean", fmt=".2e")
            lat_mean = ev.get(artifact, f"{base}/latency_wall_clock_s_mean/mean", fmt=".3f")
            lat_sd = ev.get(artifact, f"{base}/latency_wall_clock_s_mean/sd_across_seeds", fmt=".3f")
            if artifact == FOLDED_ARTIFACT:
                done = ev.get(artifact, f"{base}/n_completed")
                total = ev.get(artifact, f"{base}/n_patient_evaluations")
                completed = f"{done}/{total}"
                n_total += done
            else:
                completed = f"{len(n_seeds) * n_per_seed}/{len(n_seeds) * n_per_seed}"
                n_total += len(n_seeds) * n_per_seed
            rows.append([arm, str(degree), completed, agree, mae, maxerr, f"{lat_mean} $\\pm$ {lat_sd}"])
            raw.append([float(agree), float(mae), float(maxerr), float(lat_mean)])
    for col, idx_raw in ((3, 0), (4, 1), (5, 2), (6, 3)):
        for i in best_idx([r[idx_raw] for r in raw], higher=(idx_raw == 0)):
            rows[i][col] = bf(rows[i][col])
    n_evals = ev.literal(n_total, reason="sum of the per-row counts of completed evaluations, from the artifacts' own n and n_completed")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Real-versus-simulated CKKS validation: execution on TenSEAL/SEAL ciphertexts "
        rf"compared with the circuit-faithful {SIMULATED} evaluator, {len(n_seeds)} seeds "
        rf"$\times$ {n_per_seed} test patients per degree, label {LABEL_DISPLAY.get(label, label.removeprefix("label_").replace("_", " "))}, for the original and the "
        r"coefficient-folding polynomial evaluator. The real executions reproduced the simulated margins within the "
        r"tested scope. Best value per column in bold (agreement is bold only where it differs between rows). "
        rf"{K9_NOTE_SHORT} {PROXY}.}}",
        r"\label{tab:t5_real_vs_simulated}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Evaluator & $k$ & Completed & Agreement & Norm.\ MAE & Max abs.\ error & Latency (s) \\",
        r"\midrule",
    ]
    prev = rows[0][0]
    for r in rows:
        if r[0] != prev:
            lines.append(r"\midrule")
            prev = r[0]
        lines.append(" & ".join(r) + " \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[2pt]",
        rf"{{\footnotesize Total real-ciphertext evaluations: {n_evals}.}}",
        r"\end{table}",
    ]
    finish("t5_real_vs_simulated", "t5_real_vs_simulated.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T10 -- Full test split on real ciphertexts (revision E2)
# ---------------------------------------------------------------------------------------------
def table_t10_real_full_split() -> None:
    ev = Evidence("t10_real_full_split")
    labels = ev.get(E2_ARTIFACT, "/configuration/labels")
    n_stays = ev.get(E2_ARTIFACT, "/configuration/n_test_stays", fmt=",")
    seed = ev.get(E2_ARTIFACT, "/configuration/seed")
    degree = ev.get(E2_ARTIFACT, "/configuration/k")
    n_evals = ev.get(E2_ARTIFACT, "/n_evaluations", fmt=",")
    n_fail = ev.get(E2_ARTIFACT, "/n_failed")
    tol = ev.get(E2_ARTIFACT, "/tolerance")
    rows = []
    for lab in labels:
        base = f"/per_label/{lab}"
        rows.append([tex_escape(lab.replace("label_", "").replace("_", " ")),
                     ev.get(E2_ARTIFACT, f"{base}/f1_plaintext", fmt=".4f"),
                     ev.get(E2_ARTIFACT, f"{base}/f1_simulated", fmt=".4f"),
                     ev.get(E2_ARTIFACT, f"{base}/f1_real", fmt=".4f"),
                     ev.get(E2_ARTIFACT, f"{base}/n_sign_disagree_real_vs_simulated", fmt=","),
                     ev.get(E2_ARTIFACT, f"{base}/n_decision_disagree_real_vs_simulated", fmt=","),
                     ev.get(E2_ARTIFACT, f"{base}/max_abs_margin_error_real_vs_simulated", fmt=".2e")])
    mf = {k: ev.get(E2_ARTIFACT, f"/macro_f1/{k}", fmt=".4f") for k in ("plaintext", "simulated", "real")}
    d_real = ev.get(E2_ARTIFACT, "/delta_f1_real_vs_plaintext/point_estimate", fmt=".5f")
    d_lo = ev.get(E2_ARTIFACT, "/delta_f1_real_vs_plaintext/ci_95_lower", fmt=".5f")
    d_hi = ev.get(E2_ARTIFACT, "/delta_f1_real_vs_plaintext/ci_95_upper", fmt=".5f")
    d_sim = ev.get(E2_ARTIFACT, "/delta_f1_simulated_vs_plaintext/point_estimate", fmt=".5f")
    max_err = ev.get(E2_ARTIFACT, "/agreement_all_evaluations/max_abs_margin_error_real_vs_simulated", fmt=".2e")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Full test split on real ciphertexts: {n_stays} test stays and five labels ({n_evals} real "
        rf"evaluations, {n_fail} failed), models of training seed {seed}, degree $k={degree}$, original evaluator, "
        r"frozen validation-tuned thresholds. F1 per label for the plaintext model, the circuit-faithful "
        rf"{SIMULATED} evaluator, and real TenSEAL/SEAL ciphertexts; counts of evaluations on which the real and simulated "
        r"margins differ in sign or in the decision at the threshold; largest absolute margin error. "
        rf"The macro-F1 deviation of the real execution from plaintext is {d_real} (95\% CI [{d_lo}, {d_hi}], paired bootstrap), "
        rf"against {d_sim} for the simulator and a tolerance of {tol}. One seed and one degree. {PROXY}.}}",
        r"\label{tab:t10_real_full_split}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Label & F1 plaintext & F1 simulated & F1 real & Sign differs & Decision differs & Max abs.\ error \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(" & ".join(r) + " \\\\")
    lines += [r"\midrule",
              rf"Macro-F1 & {mf['plaintext']} & {mf['simulated']} & {mf['real']} & & & {max_err} \\",
              r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t10_real_full_split", "t10_real_full_split.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T6 -- Tree-count and tree-depth scaling (utility, latency, memory)
# ---------------------------------------------------------------------------------------------
def table_t6_scaling() -> None:
    ev = Evidence("t6_scaling")
    tolerance = ev.literal("0.03", reason="notation.MAX_ACCEPTABLE_F1_LOSS, the locked utility tolerance")
    configs = ev.get("m1_scaling_multiseed.json", "/configurations")
    degree = ev.get("m1_scaling_multiseed.json", "/k")
    label = ev.get("m1_scaling_multiseed.json", "/label")
    memory_rows = {(r["n_estimators"], r["max_depth"]): r
                    for r in ev.series("m1_scaling_memory.json", "/per_configuration/*")}

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Tree-count and tree-depth scaling at $k={degree}$, label {LABEL_DISPLAY.get(label, label.removeprefix("label_").replace("_", " "))}, "
        rf"5 seeds. $|\Delta$F1$|$ ({SIMULATED}) mean $\pm$ SD across seeds, against the 0.03 "
        r"tolerance; no cell reaches it. Latency and peak evaluation-phase memory were measured on real "
        r"ciphertexts, one fresh process per configuration. " + PROXY + r".}",
        r"\label{tab:t6_scaling}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Trees & Depth & $|\Delta$F1$|$ & Real latency (s) & Eval.\ memory (MiB) & Leaves \\",
        r"\midrule",
    ]
    for i, cfg in enumerate(configs):
        n_trees, max_depth = cfg["n_estimators"], cfg["max_depth"]
        dF1_mean = ev.get("m1_scaling_multiseed.json", f"/aggregated_across_seeds/{i}/abs_delta_f1/mean", fmt=".4f")
        dF1_sd = ev.get("m1_scaling_multiseed.json", f"/aggregated_across_seeds/{i}/abs_delta_f1/sd_across_seeds", fmt=".4f")
        lat_mean = ev.get("m1_scaling_multiseed.json", f"/aggregated_across_seeds/{i}/real_latency_wall_clock_s/mean", fmt=".3f")
        lat_sd = ev.get("m1_scaling_multiseed.json", f"/aggregated_across_seeds/{i}/real_latency_wall_clock_s/sd_across_seeds", fmt=".3f")
        n_leaves = ev.get("m1_scaling_multiseed.json", f"/aggregated_across_seeds/{i}/n_leaves_mean", fmt=".0f")
        mem_row = memory_rows.get((n_trees, max_depth))
        mem_mib = f"{mem_row['memory_kb']['evaluation_phase_increment_over_pre_evaluation_rss'] / 1024:.0f}" if mem_row else "n/a"
        lines.append(f"{n_trees} & {max_depth} & {dF1_mean} $\\pm$ {dF1_sd} & {lat_mean} $\\pm$ {lat_sd} & {mem_mib} & {n_leaves} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t6_scaling", "t6_scaling.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T7 -- Error decomposition
# ---------------------------------------------------------------------------------------------
def table_t7_error_decomposition() -> None:
    ev = Evidence("t7_error_decomposition")
    k9_note_registered = ev.literal(K9_NOTE, reason="fixed descriptive text; depth counts are separately "
                                    "recorded via m1_depth_probe.json in Table 4")
    real_degrees_note = ev.literal("3,5,7", reason="fixed descriptive text naming the real-executed degrees "
                                   "(a subset of the locked family); individually recorded via "
                                   "m1_encrypted_multiseed.json rows below")
    a_agree = ev.reduce("mean", "m1_tfhe_calibration.json", "/quantization_effect_full_grid/*/agreement_sign", fmt=".4f",
                        note=f"{SIMULATED} (float->6-bit-quantized, clear). {PROXY}.")
    a_mae = ev.reduce("mean", "m1_tfhe_calibration.json", "/quantization_effect_full_grid/*/normalized_mae", fmt=".5f")
    a_df1 = ev.reduce("mean", "m1_tfhe_calibration.json", "/quantization_effect_full_grid/*/delta_f1_quantization_only", fmt=".5f")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Error decomposition. A: float$\to$quantized (clear, 25 models). B: "
        rf"float$\to$CKKS ({SIMULATED} for all $k$, including $k=9$, which is simulation only here). "
        r"B$'$: simulated$\to$real CKKS (real ciphertexts, original evaluator, $k\in\{3,5,7\}$; "
        r"Table~\ref{tab:t5_real_vs_simulated} adds the folding evaluator and $k=9$). "
        r"A and B use different populations and metrics and are not additive. No quantized$\to$CKKS "
        r"stage exists: the CKKS pipeline never quantizes and the TFHE pipeline never uses the polynomial "
        r"approximation. " + PROXY + r".}",
        r"\label{tab:t7_error_decomposition}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Step & $k$ & Agreement & Norm.\ MAE & $\Delta$F1 \\",
        r"\midrule",
        f"A: float $\\to$ quantized & -- & {a_agree} & {a_mae} & {a_df1} \\\\",
        r"\midrule",
    ]
    for degree in (3, 5, 7, 9):
        key = f"k={degree}"
        agree = ev.reduce("mean", "m1_ckks_sweep_simulated.json", f"/per_label_detail/*/{key}/agreement", fmt=".4f")
        mae = ev.reduce("mean", "m1_ckks_sweep_simulated.json", f"/per_label_detail/*/{key}/mae_normalized", fmt=".4f")
        df1 = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/delta_f1_mean_over_seeds", fmt=".5f")
        lines.append(f"B: float $\\to$ CKKS & {degree} & {agree} & {mae} & {df1} \\\\")
    lines.append(r"\midrule")
    for degree in (3, 5, 7):
        key = f"k={degree}"
        agree = ev.get("m1_encrypted_multiseed.json", f"/pooled_across_seeds/{key}/agreement_real_vs_simulated/mean", fmt=".4f")
        mae = ev.get("m1_encrypted_multiseed.json", f"/pooled_across_seeds/{key}/normalized_mae_real_vs_simulated/mean", fmt=".2e")
        lines.append(f"B$'$: simulated $\\to$ real CKKS & {degree} & {agree} & {mae} & -- \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t7_error_decomposition", "t7_error_decomposition.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T8 -- Security and resource analysis
# ---------------------------------------------------------------------------------------------
def table_t8_security_resource() -> None:
    ev = Evidence("t8_security_resource")
    est_commit = ev.get("m1_lwe_security_summary.json", "/estimator_commit")
    matzov_bits = ev.get("m1_lwe_security_summary.json",
                          "/models/MATZOV (estimator default)/locked_chain_bits", fmt=".1f")
    matzov_meets = ev.get("m1_lwe_security_summary.json",
                           "/models/MATZOV (estimator default)/locked_chain_meets_128")
    coresvp_bits = ev.get("m1_lwe_security_summary.json",
                           "/models/ADPS16 classical core-SVP (HE Standard basis)/locked_chain_bits", fmt=".1f")
    coresvp_scale = ev.get("m1_lwe_security_summary.json",
                            "/models/ADPS16 classical core-SVP (HE Standard basis)/on_standard_scale")
    galois_mb = ev.get("m1_latency_breakdown.json", "/by_k/k=5/key_sizes_bytes/galois_keys_bytes", fmt=",")
    ctx_mb = ev.get("m1_latency_breakdown.json", "/by_k/k=5/key_sizes_bytes/evaluation_context_with_galois_keys_bytes", fmt=",")
    input_ct_mb = ev.get("m1_latency_breakdown.json", "/by_k/k=5/ciphertext_bytes/input_total_per_patient", fmt=",")
    peak_rss = ev.reduce("max", "m1_scaling_memory.json", "/per_configuration/*/memory_kb/peak_rss_during_evaluation_phase", fmt=",")
    est_commit_short = ev.literal(est_commit[:10], reason="10-character abbreviation of the full estimator "
                                   "commit hash already recorded above; a common, readable display convention")
    sec_target = ev.literal(128, reason="notation.LAM_SECURITY_LEVEL, the locked target security level")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Security and resource analysis. Independent LWE estimate: lattice-estimator "
        rf"(Albrecht, Player, Scott), commit \texttt{{{est_commit_short}}}. The {sec_target}-bit level holds "
        rf"under the estimator's default MATZOV cost model, which reproduces the HE Security Standard's "
        r"published rows within 5 bits. It is not claimed under every cost model: raw core-SVP scores the "
        r"chain lower and is not on the Standard's scale. " + PROXY + r".}",
        r"\label{tab:t8_security_resource}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Quantity & Value \\",
        r"\midrule",
        f"Security, MATZOV (estimator default) & {matzov_bits} bits (meets 128: {matzov_meets}) \\\\",
        f"Security, raw core-SVP (ADPS16) & {coresvp_bits} bits (on Standard's scale: {coresvp_scale}; "
        r"not compared with 128) \\\\",
        r"SEAL TC128 enforcement & satisfied (Table~\ref{tab:t3_ckks_circuit_config}) \\",
        f"Galois (rotation) keys, $k=5$ & {galois_mb} bytes \\\\",
        f"Evaluation context with keys, $k=5$ & {ctx_mb} bytes \\\\",
        f"Input ciphertexts per patient, $k=5$ & {input_ct_mb} bytes \\\\",
        f"Peak evaluation-phase RSS (max over scaling grid) & {peak_rss} KiB \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    finish("t8_security_resource", "t8_security_resource.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# T9 -- Comparison with existing work (guide Sec 26). Source data: paper_assets/related_work.json,
# one entry per source in M1_Literature_Map.md. Cells are truncated for the printed table (12
# columns at full IEEE two-column width leaves little room per cell); the untruncated value is
# still recorded in each Evidence.external() record's `note`, and the full text lives in
# related_work.json and M1_Literature_Map.md -- nothing is lost, only the printed cell is terse.
# ---------------------------------------------------------------------------------------------
def _short(text: str, maxlen: int = 20) -> str:
    # Plain ASCII "..." rather than a Unicode ellipsis -- the vendored IEEEtran/bibtex toolchain's
    # UTF-8 handling is unreliable outside inputenc's covered range (see m1.bib's ASCII-only note);
    # sticking to ASCII here avoids re-litigating that per generated table.
    text = str(text)
    if len(text) <= maxlen:
        return text
    return text[: maxlen - 3].rstrip() + "..."


def table_t9_related_work() -> None:
    ev = Evidence("t9_related_work")
    with open(RELATED_WORK_JSON, "r", encoding="utf-8") as f:
        papers = json.load(f)["papers"]

    # Printed columns: the five fields most sources report. Depth, packing, latency, memory and
    # utility were reported by at most a few sources; they stay in related_work.json, and this
    # work's own values are in Tables 4, 6 and 8.
    shown = [("model", "model_type"), ("scheme", "he_scheme"), ("security", "security_level"),
             ("degree", "polynomial_degree"), ("execution", "real_or_simulated")]

    rows = []
    for p in papers:
        short_name = ev.literal(p["short_name"], reason="citation display label (author(s) + year), not a measured value")
        cells = [tex_escape(short_name) + rf"~\cite{{{p['citation']}}}"]
        for disp_key, field in shown:
            displayed = ev.external(p["display"][disp_key], citation=p["citation"], locus=p["locus"],
                                     note=f"abbreviation of the recorded field '{field}': {p[field]}")
            cells.append(tex_escape(displayed))
        rows.append(cells)

    degree = ev.literal("3, 5, 7, 9 (Chebyshev comparison)", reason="notation.K_APPROX_DEGREE_VALUES, the evaluated degree family")
    sec_bits = ev.literal(128, reason="notation.LAM_SECURITY_LEVEL; SEAL TC128 check (Table 3/8)")
    n_trees = ev.get("m1_plaintext_baseline.json", "/n_estimators")
    max_depth = ev.get("m1_plaintext_baseline.json", "/max_depth")
    this_work = [r"\textbf{This work}", f"XGBoost ({n_trees} trees, depth {max_depth})", "CKKS (SEAL/TenSEAL)",
                 f"{sec_bits}-bit (SEAL check)", degree,
                 r"Real, all four degrees ($k=9$ with the folding evaluator); utility simulated, real at the selected degree"]

    header = ["Source", "Model", "Scheme", "Security", "Comparison degree", "Execution"]
    for w in ("0.20", "0.17", "0.09"):
        ev.literal(w, reason="column width as a fraction of the text width (layout, not data)")
    rr = r">{\raggedright\arraybackslash}"
    col_spec = (rf"{rr}p{{0.20\textwidth}}{rr}p{{0.20\textwidth}}{rr}p{{0.17\textwidth}}"
                rf"{rr}p{{0.09\textwidth}}{rr}p{{0.17\textwidth}}{rr}p{{0.09\textwidth}}")

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Comparison with existing encrypted-inference work: {len(papers)} sources, each verified "
        r"against a primary record. ``NR'' = not reported by that source, never estimated; ``n/a'' = not "
        r"applicable. Entries are abbreviated from the recorded fields, which are in the "
        r"reproducibility package together with multiplicative depth, packing, latency, memory and "
        r"utility (reported by few sources). This work's depth, latency and memory are in "
        r"Tables~\ref{tab:t4_degree_grid}, \ref{tab:t6_scaling} and~\ref{tab:t8_security_resource}. " + PROXY + r".}",
        r"\label{tab:t9_related_work}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(header) + r" \\",
        r"\midrule",
    ]
    for cells in rows:
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join(this_work) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t9_related_work", "t9_related_work.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# Supplementary tables (not in the guide's numbered 9, kept as reference material)
# ---------------------------------------------------------------------------------------------
def table_s1_k_selection() -> None:
    ev = Evidence("s1_k_selection")
    tolerance = ev.literal("0.03", reason="notation.MAX_ACCEPTABLE_F1_LOSS, the locked utility tolerance")
    k9_note_registered = ev.literal(K9_NOTE, reason="fixed descriptive text; depth counts are separately "
                                    "recorded via m1_depth_probe.json in Table 4")
    k_star = ev.get("m1_k_selection.json", "/k_star")
    feasible = ev.get("m1_k_selection.json", "/feasible_set")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Cheapest-feasible-degree selection under the pre-specified rule "
        r"minimize $L(k)$ s.t.\ $\Delta$F1$(k)\le 0.03$. "
        r"The selected $k^*$ is the lowest degree of the evaluated family in "
        r"every tested seed and label, so the utility constraint never binds "
        r"(Table~\ref{tab:t4_degree_grid}). This is not an adaptive-selection result. " + PROXY + r".}",
        r"\label{tab:s1_k_selection}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"$k$ & $\Delta$F1 & Latency (s) & Feasible \\",
        r"\midrule",
    ]
    for i, e in enumerate(feasible):
        lat_mean = ev.get("m1_k_selection.json", f"/feasible_set/{i}/online_latency_seconds_per_patient", fmt=".2f") \
            if e["online_latency_seconds_per_patient"] is not None else None
        lat_sd = ev.get("m1_k_selection.json", f"/feasible_set/{i}/latency_sd_across_seeds", fmt=".2f") \
            if e["online_latency_seconds_per_patient"] is not None else None
        lat = f"{lat_mean} $\\pm$ {lat_sd}" if lat_mean is not None else "n/a$^\\dagger$"
        dF1 = ev.get("m1_k_selection.json", f"/feasible_set/{i}/delta_f1_mean", fmt=".4f")
        deg = ev.get("m1_k_selection.json", f"/feasible_set/{i}/k")
        star = " $\\leftarrow k^*$" if deg == k_star else ""
        lines.append(f"{deg} & {dF1} & {lat} & \\checkmark{star} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}", r"\\[2pt]",
        rf"{{\footnotesize $\dagger$ {K9_NOTE}.}}", r"\end{table}",
    ]
    finish("s1_k_selection", "s1_k_selection.tex", lines, ev)


def table_s2_tfhe_calibration() -> None:
    ev = Evidence("s2_tfhe_calibration")
    n_bits = ev.get("m1_tfhe_calibration.json", "/freeze_list/n_bits")
    n_exec = ev.get("m1_tfhe_calibration.json", "/real_fhe_execution/n_real_fhe_executions")
    a_agree = ev.reduce("mean", "m1_tfhe_calibration.json", "/quantization_effect_full_grid/*/agreement_sign", fmt=".4f")
    a_mae = ev.reduce("mean", "m1_tfhe_calibration.json", "/quantization_effect_full_grid/*/normalized_mae", fmt=".5f")
    tfhe_agree = ev.get("m1_tfhe_calibration.json", "/real_fhe_execution/agreement_tfhe_vs_quantized_clear", fmt=".4f")
    tfhe_lat_mean = ev.get("m1_tfhe_calibration.json", "/real_fhe_execution/online_latency_seconds_mean", fmt=".2f")
    tfhe_lat_std = ev.get("m1_tfhe_calibration.json", "/real_fhe_execution/online_latency_seconds_std", fmt=".2f")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Quantized-TFHE reference (Concrete-ML, "
        rf"$n_\text{{bits}}={n_bits}$), a calibration reference rather than a comparison of CKKS and "
        r"TFHE in either direction. Quantization "
        r"effect: full 5-seed$\times$5-label grid (clear simulation). Real "
        rf"encrypted execution: one representative model (seed 42, hypertension), n={n_exec}. "
        r"" + PROXY + r".}",
        r"\label{tab:s2_tfhe}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
        f"Quantization agreement (mean, 25 models) & {a_agree} \\\\",
        f"Quantization normalized MAE (mean, 25 models) & {a_mae} \\\\",
        f"Agreement(TFHE, quantized-clear) & {tfhe_agree} \\\\",
        f"Real FHE online latency (s/sample) & {tfhe_lat_mean} $\\pm$ {tfhe_lat_std} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    finish("s2_tfhe_calibration", "s2_tfhe_calibration.tex", lines, ev)


def table_s3_gate_m1a() -> None:
    ev = Evidence("s3_gate_m1a")
    arms = ev.get("m1_gate_a_report.json", "/arms")
    thresholds_note = ev.literal(
        "pre-specified thresholds agreement >= 0.95, normalized MAE <= 0.10, normalized RMSE <= 0.15, "
        "after the Amendment v1.2 gamma-recalibration correction",
        reason="declared, pre-specified gate thresholds and amendment reference (m1-0-implementation-lock.md), "
        "not values measured by any results/*.json artifact",
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Staged feasibility check on a reduced model (5 trees, depth $\le$3, "
        r"$k=5$), after the correction of the sigmoid steepness $\gamma$. Pre-specified "
        r"thresholds: agreement $\ge 0.95$, normalized MAE $\le 0.10$, normalized RMSE $\le 0.15$. "
        f"{SIMULATED}. " + PROXY + r".}",
        r"\label{tab:s3_gate_m1a}",
        r"\begin{tabular}{lrrrl}",
        r"\toprule",
        r"Arm & Agreement & Norm.\ MAE & Norm.\ RMSE & Pass \\",
        r"\midrule",
    ]
    for i, arm in enumerate(arms):
        name = ev.get("m1_gate_a_report.json", f"/arms/{i}/arm").replace("_", " ")
        agree = ev.get("m1_gate_a_report.json", f"/arms/{i}/agreement_sign", fmt=".4f")
        mae = ev.get("m1_gate_a_report.json", f"/arms/{i}/normalized_mae", fmt=".4f")
        rmse = ev.get("m1_gate_a_report.json", f"/arms/{i}/normalized_rmse", fmt=".4f")
        passed = ev.get("m1_gate_a_report.json", f"/arms/{i}/gate_pass")
        pass_mark = "\\checkmark" if passed else "$\\times$"
        lines.append(f"{name} & {agree} & {mae} & {rmse} & {pass_mark} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("s3_gate_m1a", "s3_gate_m1a.tex", lines, ev)


def table_s4_decision_agreement() -> None:
    """Added analysis S2.1 (docs/step2_prespec_m1.md section B): decision disagreement between plaintext and
    simulated CKKS at the frozen per-(seed, label) thresholds, per label and degree, on the full test split."""
    ev = Evidence("s4_decision_agreement")
    art = "m1_decision_agreement.json"
    ev.get(art, "/backend", note=f"{SIMULATED}. {PROXY}. Added analysis (decision-level fidelity). {K9_NOTE}.")
    labels = ev.get(art, "/labels")
    degrees = ev.get(art, "/degrees")
    cells = ev.series(art, "/cells/*")
    index = {(c["label"], c["k"]): i for i, c in enumerate(cells)}
    n_test = ev.get(art, "/n_test_stays", fmt=",")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Added analysis: decision disagreement between the plaintext model and the {SIMULATED} CKKS circuit, "
        rf"as a fraction of the {n_test} test stays, at the frozen per-label decision thresholds. Mean $\pm$ SD "
        r"over the five training seeds (training variability). The last row is the mean over the five labels with a "
        r"patient-cluster bootstrap 95\% interval (sampling uncertainty), reported separately. Below: mean absolute "
        r"difference of the predicted probability and mean AUROC difference (plaintext minus CKKS). "
        r"$k=1$ is the additional diagnostic arm. " + PROXY + r".}",
        r"\label{tab:s4_decision_agreement}",
        r"\begin{tabular}{l" + "r" * len(degrees) + "}",
        r"\toprule",
        "Label & " + " & ".join(f"$k={d}$" for d in degrees) + r" \\",
        r"\midrule",
    ]
    for label in labels:
        row = [tex_escape(LABEL_DISPLAY.get(label, label.removeprefix("label_").replace("_", " ")))]
        for d in degrees:
            i = index[(label, d)]
            m = ev.get(art, f"/cells/{i}/disagree_rate_mean", fmt=".4f")
            sd = ev.get(art, f"/cells/{i}/disagree_rate_sd_across_seeds", fmt=".4f")
            row.append(f"{m} $\pm$ {sd}")
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")
    mean_row, ci_row = ["Mean over labels"], ["Bootstrap 95\% interval"]
    for d in degrees:
        m = ev.get(art, f"/by_k/k={d}/mean_over_labels_and_seeds", fmt=".4f")
        sd = ev.get(art, f"/by_k/k={d}/sd_across_seeds_of_label_mean", fmt=".4f")
        lo = ev.get(art, f"/by_k/k={d}/cluster_bootstrap_ci95_of_label_seed_mean/0", fmt=".4f")
        hi = ev.get(art, f"/by_k/k={d}/cluster_bootstrap_ci95_of_label_seed_mean/1", fmt=".4f")
        mean_row.append(f"{m} $\pm$ {sd}")
        ci_row.append(f"[{lo}, {hi}]")
    lines.append(" & ".join(mean_row) + r" \\")
    lines.append(" & ".join(ci_row) + r" \\")
    lines.append(r"\midrule")
    pe_row, au_row = ["Mean $|\\Delta p|$ (probability)"], ["Mean AUROC difference"]
    for d in degrees:
        pe_row.append(ev.get(art, f"/by_k/k={d}/prob_abs_error_mean_over_labels_and_seeds", fmt=".4f"))
        au_row.append(ev.get(art, f"/by_k/k={d}/auroc_diff_plain_minus_ckks_mean_over_labels_and_seeds", fmt="+.5f"))
    lines.append(" & ".join(pe_row) + r" \\")
    lines.append(" & ".join(au_row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("s4_decision_agreement", "s4_decision_agreement.tex", lines, ev)


def table_s5_interleaved_latency() -> None:
    """Added analysis S2.3a: original and folding evaluators re-timed in interleaved order, two rounds."""
    ev = Evidence("s5_interleaved_latency")
    art = "m1_interleaved_latency_summary.json"
    ev.get(art, "/protocol", note=f"REAL ciphertext execution. {PROXY}. Added analysis (interleaved re-timing). {K9_NOTE}.")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Added analysis: per-query latency (s) of the original and the coefficient-folding evaluator re-timed in "
        r"interleaved order on one day (round 2 in the opposite order to round 1; one label's 50-tree, depth-6 model; "
        r"20 timed queries per cell, mean $\pm$ SD over queries). The last column is the folding-to-original ratio, mean of "
        r"the two rounds. " + K9_NOTE_SHORT + " " + PROXY + r".}",
        r"\label{tab:s5_interleaved_latency}",
        r"\begin{tabular}{lcccc|c}",
        r"\toprule",
        r"$k$ & Original, round 1 & Original, round 2 & Folding, round 1 & Folding, round 2 & Ratio \\",
        r"\midrule",
    ]
    for k in (3, 5, 7, 9):
        cells = []
        for evaluator in ("original", "folding"):
            for rnd in (1, 2):
                if k == 9 and evaluator == "original":
                    cells.append("n/a$^\\dagger$")
                    continue
                m = ev.get(art, f"/by_k/k={k}/{evaluator}/round_{rnd}_mean_s", fmt=".3f")
                sd = ev.get(art, f"/by_k/k={k}/{evaluator}/round_{rnd}_sd_s", fmt=".3f")
                cells.append(f"{m} $\\pm$ {sd}")
        ratio = ev.get(art, f"/by_k/k={k}/folding_over_original_ratio_mean", fmt=".2f") if k != 9 else "n/a"
        lines.append(f"{k} & " + " & ".join(cells) + f" & {ratio} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]", rf"{{\footnotesize $\dagger$ {K9_NOTE}.}}", r"\end{table}"]
    finish("s5_interleaved_latency", "s5_interleaved_latency.tex", lines, ev)


def table_s6_ps_benchmark() -> None:
    """Added analysis S2.4: the polynomial step on one ciphertext through both SEAL evaluators and OpenFHE's
    Paterson-Stockmeyer Chebyshev evaluation."""
    ev = Evidence("s6_ps_benchmark")
    seal, ofhe = "m1_seal_scalar_poly_benchmark.json", "m1_openfhe_ps_benchmark.json"
    ev.get(ofhe, "/backend", note=f"REAL OpenFHE and SEAL ciphertext execution. {PROXY}. Added analysis (library comparison). {K9_NOTE}.")
    n_slots = ev.get(ofhe, "/n_slots", fmt=",")
    ring = ev.get(ofhe, "/context/ring_dim", fmt=",")
    sec = ev.literal("128", reason="the 128-bit classical security level (HEStd_128_classic) of the OpenFHE context, "
                     "recorded as context.security in m1_openfhe_ps_benchmark.json")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Added analysis: the polynomial step on one ciphertext of {n_slots} slots, same comparison polynomial, "
        r"through the two SEAL evaluators and OpenFHE's Paterson--Stockmeyer Chebyshev evaluation (levels consumed; "
        r"time in s, mean over 20 timed repetitions). SEAL levels are the polynomial depths of the circuit's evaluators; "
        r"the OpenFHE levels were read from the library. ``+ per-slot'' adds the per-slot affine step that the circuit's "
        r"per-slot coefficients need (one more level). The libraries differ in RNS layout, rescaling and key switching "
        rf"(OpenFHE: hybrid key switching, one digit per tower, ring degree {ring} at {sec}-bit security), so this compares "
        r"libraries, not evaluators inside one. " + K9_NOTE_SHORT + " " + PROXY + r".}",
        r"\label{tab:s6_ps_benchmark}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{SEAL original} & \multicolumn{2}{c}{SEAL folding} & \multicolumn{2}{c}{OpenFHE PS} & "
        r"\multicolumn{2}{c}{OpenFHE PS + per-slot} \\",
        r"$k$ & levels & time & levels & time & levels & time & levels & time \\",
        r"\midrule",
    ]
    for k in (3, 5, 7, 9):
        lv_o = ev.get("m1_depth_probe.json", f"/per_degree/k={k}/code_derived/polynomial_depth")
        lv_f = ev.get("m1_folded_depth_accounting.json", f"/per_degree/k={k}/polynomial_depth_folded")
        t_o = ev.get(seal, f"/by_k/k={k}/original/mean_s", fmt=".3f") if k != 9 else "n/a$^\\dagger$"
        t_f = ev.get(seal, f"/by_k/k={k}/folding/mean_s", fmt=".3f")
        ps_l = ev.get(ofhe, f"/by_k/k={k}/levels_consumed_scalar")
        ps_t = ev.get(ofhe, f"/by_k/k={k}/scalar_polynomial/mean_s", fmt=".3f")
        pa_l = ev.get(ofhe, f"/by_k/k={k}/levels_consumed_with_per_slot_affine")
        pa_t = ev.get(ofhe, f"/by_k/k={k}/with_per_slot_affine/mean_s", fmt=".3f")
        lines.append(f"{k} & {lv_o} & {t_o} & {lv_f} & {t_f} & {ps_l} & {ps_t} & {pa_l} & {pa_t} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\\[2pt]", rf"{{\footnotesize $\dagger$ {K9_NOTE}.}}", r"\end{table}"]
    finish("s6_ps_benchmark", "s6_ps_benchmark.tex", lines, ev)


# ---------------------------------------------------------------------------------------------
# Continuous-feature task (added analysis): T11 fidelity, T12 utility, T13 real-ciphertext check, S7 pilot trajectory
# ---------------------------------------------------------------------------------------------
CT_SUMMARY = "m1c_final_summary.json"
CT_NOTE = (f"{PROXY}. Continuous-feature task (triage vital signs, age, triage acuity). Added analysis. "
           f"{SIMULATED} unless stated; test split, five training seeds.")


def _pm(ev: Evidence, art: str, ptr: str, fmt: str) -> str:
    return f"{ev.get(art, ptr + '/mean', fmt=fmt)} $\\pm$ {ev.get(art, ptr + '/sd', fmt=fmt)}"


def _pm_pct(ev: Evidence, art: str, ptr: str) -> str:
    m, sd = ev.get(art, ptr + '/mean', fmt='.1%'), ev.get(art, ptr + '/sd', fmt='.1%')
    return f"{m} $\\pm$ {sd}".replace("%", "\\%")


def table_t11_continuous_fidelity() -> None:
    ev = Evidence("t11_continuous_fidelity")
    art = CT_SUMMARY
    unit = ev.literal("1,000", reason="unit of the rate, decisions per 1,000 stays")
    ev.get(art, "/source", note=CT_NOTE)
    n_test = ev.get(art, "/n_test_stays", fmt=",")
    n_seeds = ev.get(art, "/n_seeds")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Continuous-feature task: fidelity of the {SIMULATED} CKKS circuit to its own plaintext model on the {n_test} test stays, "
        rf"at the F1-maximizing decision threshold. Mean $\pm$ SD over {n_seeds} training seeds (training variability); "
        r"patient-cluster bootstrap intervals are in the text (sampling uncertainty, reported separately). "
        r"Approximate rows: full-precision inputs, soft-step comparison polynomial of degree $k$ on a per-feature interval. "
        r"Exact rows: inputs quantized to bins, comparison polynomial interpolating the step exactly at the finitely many possible values "
        r"(degree $2B-1$ for $B$ bins); the 8-bin row is simulation only because no evaluator for degree 15 is built into the circuit. "
        r"$k=1$ is an additional diagnostic arm outside the evaluated degree family. "
        r"The 5-bin degree-9 circuit also runs on real ciphertexts (Table~\ref{tab:t13_continuous_real}). " + PROXY + r".}",
        r"\label{tab:t11_continuous_fidelity}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        rf"Comparison & Degree & Differing decisions / {unit} stays & Plaintext positives lost & Circuit AUROC \\",
        r"\midrule",
    ]
    for d in (1, 3, 5, 7, 9):
        k = f"k={d}"
        lines.append(" & ".join([
            "Approximate, full precision" if d == 1 else "", f"$k={d}$",
            _pm(ev, art, f"/floor/{k}/f1/differing_per_1000", ".1f"),
            _pm_pct(ev, art, f"/floor/{k}/f1/lost_share_of_plain_positives"),
            _pm(ev, art, f"/floor/{k}/circuit_auroc", ".4f")]) + r" \\")
    lines.append(r"\midrule")
    for label, key in (("Exact, 5 bins", "exact_B5"), ("Exact, 8 bins (simulation only)", "exact_B8")):
        deg = ev.get(art, f"/{key}/degree")
        lines.append(" & ".join([
            label, f"$k={deg}$",
            _pm(ev, art, f"/{key}/f1/differing_per_1000", ".1f"),
            _pm_pct(ev, art, f"/{key}/f1/lost_share_of_plain_positives"),
            _pm(ev, art, f"/{key}/circuit_auroc", ".4f")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t11_continuous_fidelity", "t11_continuous_fidelity.tex", lines, ev)


def table_t12_continuous_utility() -> None:
    ev = Evidence("t12_continuous_utility")
    art = CT_SUMMARY
    ev.get(art, "/source", note=CT_NOTE)
    n_test = ev.get(art, "/n_test_stays", fmt=",")
    n_seeds = ev.get(art, "/n_seeds")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Continuous-feature task: predictive utility of the plaintext models on the {n_test} test stays. Mean $\pm$ SD over {n_seeds} "
        r"training seeds. The first row is the full-precision model; the other rows quantize every input to bins and are the models the "
        r"exact circuits evaluate. The cost of quantization is a property of the model, not of the circuit, and is reported here separately "
        r"from circuit error (Table~\ref{tab:t11_continuous_fidelity}). Columns 4 to 6: the F1-maximizing threshold set on the validation "
        r"split. Columns 7 and 8: the threshold set on the validation split to reach 80\% sensitivity. " + PROXY + r".}",
        r"\label{tab:t12_continuous_utility}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & AUROC & AUPRC & F1 & Sens. & PPV & Spec. & PPV \\",
        r"\midrule",
    ]
    for label, key in (("Full precision", "plaintext_full_precision"), ("5 bins", "plaintext_quantized_B5"),
                       ("8 bins", "plaintext_quantized_B8")):
        lines.append(" & ".join([
            label,
            _pm(ev, art, f"/{key}/auroc", ".4f"), _pm(ev, art, f"/{key}/auprc", ".4f"), _pm(ev, art, f"/{key}/f1op_f1", ".3f"),
            _pm(ev, art, f"/{key}/f1op_sensitivity", ".3f"), _pm(ev, art, f"/{key}/f1op_ppv", ".3f"),
            _pm(ev, art, f"/{key}/sens80_specificity", ".3f"), _pm(ev, art, f"/{key}/sens80_ppv", ".3f")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t12_continuous_utility", "t12_continuous_utility.tex", lines, ev)


def table_t13_continuous_real() -> None:
    ev = Evidence("t13_continuous_real")
    art = "m1c_final_real.json"
    ev.get(art, "/backend", note=f"REAL ciphertext execution (locked parameters). {PROXY}. Added analysis, continuous-feature task, test split.")
    seeds = ev.get(art, "/per_seed")
    gate = ev.get(art, "/gate_tolerance", fmt="sci2")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Continuous-feature task, 5-bin exact circuit of degree 9 on \textsc{real} ciphertexts (coefficient-folding evaluator, "
        r"locked parameters). Per seed: a frozen stratified test sample (plaintext-positive and plaintext-negative stays at the F1-maximizing "
        rf"threshold); decisions agreeing with the plaintext model; largest absolute difference between the decrypted margin and the {SIMULATED} margin "
        rf"(correctness gate {gate}); wall-clock seconds per stay for one label. " + PROXY + r".}",
        r"\label{tab:t13_continuous_real}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Seed & Stays & Agreeing decisions & Max margin difference & Seconds per stay \\",
        r"\midrule",
    ]
    for s in seeds:
        n = ev.get(art, f"/per_seed/{s}/n_stays", fmt=",")
        a = ev.get(art, f"/per_seed/{s}/decisions_agreeing_with_booster", fmt=",")
        e = ev.get(art, f"/per_seed/{s}/max_abs_real_minus_sim", fmt="sci2")
        t = ev.get(art, f"/per_seed/{s}/mean_seconds_per_stay", fmt=".2f")
        lines.append(f"{s} & {n} & {a} & {e} & {t}" + r" \\")
    lines.append(r"\midrule")
    lines.append(" & ".join([r"All", ev.get(art, "/n_stays", fmt=","), ev.get(art, "/decisions_agreeing_with_booster", fmt=","),
                             ev.get(art, "/max_abs_real_minus_sim", fmt="sci2"), ev.get(art, "/mean_seconds_per_stay", fmt=".2f")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t13_continuous_real", "t13_continuous_real.tex", lines, ev)


def table_s7_pilot_trajectory() -> None:
    """Supplementary: how the continuous-feature configuration was reached. Validation split, seed 42, simulated; every row is an
    added exploratory analysis whose pre-specification was committed before it ran (module DEVIATIONS.md)."""
    ev = Evidence("s7_pilot_trajectory")
    ev.get("m1c_pilot2.json", "/backend", note=f"{SIMULATED}. {PROXY}. Added exploratory analyses (validation split, one seed).")

    def pos(art, ptr):
        return ev.get(art, ptr, fmt=".1f")

    def pct(art, ptr):
        return ev.get(art, ptr, fmt=".1%").replace("%", "\\%")

    unit = ev.literal("1,000", reason="unit of the rate, decisions per 1,000 stays")
    dash = str(ev.literal("--", reason="share of lost positives not recorded in the earlier pilot artifact"))
    zero_lost = str(ev.literal("0\\%", reason="no decision differs in the 5-bin exact simulation, so no positive is lost"))
    rows = [
        ("Full precision, one pooled interval", "$k=9$", pos("m1c_pilot2.json", "/configs/pooled|threshold_decision/cells/k=9/disagree_per_1000"), dash),
        ("Full precision, per-feature interval", "$k=9$", pos("m1c_pilot2.json", "/configs/per_feature|threshold_decision/cells/k=9/disagree_per_1000"), dash),
        ("Full precision, per-feature interval", "$k=15$", pos("m1c_pilot3.json", "/results/soft_step/k=15/disagree_per_1000"),
         pct("m1c_pilot3.json", "/results/soft_step/k=15/lost_share_of_plain_positives")),
        ("Full precision, per-feature interval", "$k=63$", pos("m1c_pilot4.json", "/results/k=63/disagree_per_1000"),
         pct("m1c_pilot4.json", "/results/k=63/lost_share_of_plain_positives")),
        ("8 bins, approximate comparison", "$k=9$", pos("m1c_pilot6.json", "/levels/b=8/cells/k=9/disagree_per_1000"),
         pct("m1c_pilot6.json", "/levels/b=8/cells/k=9/lost_share_of_plain_positives")),
        ("8 bins, approximate comparison", "$k=15$", pos("m1c_pilot6.json", "/levels/b=8/cells/k=15/disagree_per_1000"),
         pct("m1c_pilot6.json", "/levels/b=8/cells/k=15/lost_share_of_plain_positives")),
        ("5 bins, exact comparison", "$k=9$", ev.get("m1c_pilot7.json", "/levels/B=5/sim_n_decisions_differing", fmt=".1f"), zero_lost),
    ]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Supplementary: how the continuous-feature configuration was reached. Seed 42, validation split, {SIMULATED}; exploratory "
        rf"analyses\added{{}}, each pre-specified in a committed note before it ran. Columns: decisions differing from the plaintext model per {unit} stays, and the "
        r"share of plaintext positives lost (``--'': not recorded in that artifact). The degrees $k=15$ and $k=63$ lie outside the evaluated "
        r"family and are additional diagnostic arms. All rows use the seven triage vital signs and age only, without triage acuity, "
        r"so they are not comparable in level to Table~\ref{tab:t11_continuous_fidelity}. The last row counts differing decisions, "
        r"which is zero, with no positive lost. " + PROXY + r".}",
        r"\label{tab:s7_pilot_trajectory}",
        r"\begin{tabular}{llrr}",
        r"\toprule",
        rf"Configuration & Degree & Differing / {unit} & Positives lost \\",
        r"\midrule",
    ]
    for cfg, deg, d, lost in rows:
        lines.append(f"{cfg} & {deg} & {d} & {lost}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("s7_pilot_trajectory", "s7_pilot_trajectory.tex", lines, ev)


TMS_ART = "m1c_thermo_final_summary.json"
TM_NOTE = ("Structure- and threshold-hiding thermometer protocol, final run (added study; Analysis-plan gate in GATES.md). "
           "Test split, five seeds.")


def _tm_real_row(ev: Evidence, label: str, who: str, B: int, circuit: str, chain_key: str) -> str:
    r = f"/by_B/B={B}/real/{circuit}"
    f1 = ev.get(TMS_ART, f"{r}/decisions_agreeing_f1_op", fmt=",")
    s80 = ev.get(TMS_ART, f"{r}/decisions_agreeing_sens80_op", fmt=",")
    return " & ".join([
        label, who, str(B), str(ev.get(TMS_ART, f"/chain_levels/{chain_key}")),
        ev.get(TMS_ART, f"{r}/client_upload_mb", fmt=".2f"),
        _pm(ev, TMS_ART, f"{r}/seconds_per_stay", ".2f"),
        ev.get(TMS_ART, f"{r}/max_abs_margin_error", fmt="sci2"),
        f"{f1} / {s80}"]) + r" \\"


def table_t14_thermometer_real() -> None:
    ev = Evidence("t14_thermometer_real")
    ev.get(TMS_ART, "/source", note=f"REAL ciphertext execution. {PROXY}. {TM_NOTE}")
    n_seeds = ev.get(TMS_ART, "/n_seeds")
    per_seed = ev.get(TMS_ART, "/n_real_stays_per_seed_per_B")
    n_total = ev.get(TMS_ART, "/by_B/B=5/real/packed_thermometer_4level/n_stays_total", fmt=",")
    gate = ev.literal("5\\times10^{-4}", reason="correctness gate of the signed plan, declared constant")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Exact circuits on \textsc{{real}} ciphertexts, by what the client must know. A frozen stratified test sample of {per_seed} stays per seed "
        rf"and bin count over {n_seeds} seeds (plaintext-positive and plaintext-negative at the F1-maximizing threshold). Upload is the per-stay ciphertext "
        r"size in MB (one-off keys are in Table~\ref{tab:s9_keys}); seconds are wall-clock per stay for one label, mean $\pm$ SD over seeds; the last column is "
        rf"the number of the {n_total} sampled stays whose decision agrees with the plaintext model, at the F1 point / at the 80\%-sensitivity point. Every circuit meets the correctness gate (margin error at most ${gate}$ "
        r"and every F1-point decision agreeing). Chain levels are the data levels of the coefficient chain the circuit ran on. Interpolation: degree-9 polynomial, "
        rf"coefficient-folding evaluator, locked chain. Compared with the {SIMULATED} circuits in Table~\ref{{tab:t15_thermometer_exactness}}. " + PROXY + r".}",
        r"\label{tab:t14_thermometer_real}",
        r"\begin{tabular}{llrrrrrl}",
        r"\toprule",
        r"Circuit & Client knows & $B$ & Chain levels & Upload (MB) & Seconds per stay & Max margin error & Agreeing decisions \\",
        r"\midrule",
    ]
    for B in (5, 8, 16):
        lines.append(_tm_real_row(ev, "Thermometer, broadcast" if B == 5 else "", "nothing, if it follows the protocol" if B == 5 else "", B,
                                  "broadcast_thermometer_4level", "four_level"))
    for B in (5, 8, 16):
        lines.append(_tm_real_row(ev, "Thermometer, packed" if B == 5 else "", "nothing, if it follows the protocol" if B == 5 else "", B,
                                  "packed_thermometer_4level", "four_level"))
    lines.append(r"\midrule")
    lines.append(_tm_real_row(ev, "Exact interpolation", "the thresholds", 5, "exact_interpolation_7level", "locked_seven_level"))
    lines.append(r"\midrule")
    for B in (5, 8, 16):
        lines.append(_tm_real_row(ev, "Indicators computed by client" if B == 5 else "", "structure and thresholds" if B == 5 else "", B,
                                  "client_indicator_baseline_4level", "four_level"))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t14_thermometer_real", "t14_thermometer_real.tex", lines, ev)


def table_t15_thermometer_exactness() -> None:
    ev = Evidence("t15_thermometer_exactness")
    ev.get(TMS_ART, "/source", note=f"{SIMULATED} circuit on the full test split plus REAL ciphertexts for the near-threshold arm. {PROXY}. {TM_NOTE}")
    n_test = ev.get(TMS_ART, "/n_test_stays", fmt=",")
    n_seeds = ev.get(TMS_ART, "/n_seeds")
    reps_big = ev.get(TMS_ART, "/by_B/B=5/near_threshold/f1/repetitions_per_seed/0")
    reps_small = ev.get(TMS_ART, "/by_B/B=5/near_threshold/f1/repetitions_per_seed/1")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Thermometer circuit: utility, exactness and stays within CKKS noise of a threshold, on the {n_test} test stays, {n_seeds} seeds. AUROC and F1: mean $\pm$ SD over seeds "
        r"(training variability; the cost is the full-precision AUROC minus the quantized AUROC, a property of the model and not of the circuit). "
        rf"Differing decisions: the {SIMULATED} circuit against the plaintext model, summed over seeds, at the F1 point / 80\%-sensitivity point. "
        r"Near-threshold stays\added{}: test stays whose plaintext margin lies within the correctness gate of a threshold, evaluated on \textsc{real} ciphertexts "
        rf"with {reps_big} fresh encryptions each for seed 42 and {reps_small} for the other seeds; the count is stays flipped in at least one repetition out of stays, summed over seeds, "
        r"at the F1 threshold / 80\%-sensitivity threshold. " + PROXY + r".}",
        r"\label{tab:t15_thermometer_exactness}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Model & AUROC & AUROC cost & F1 & Differing decisions & Max margin difference & Near-threshold: flipped / stays \\",
        r"\midrule",
        " & ".join(["Full precision", _pm(ev, TMS_ART, "/full_precision/auroc_test", ".4f"), "---",
                     _pm(ev, TMS_ART, "/full_precision/f1_f1_op", ".3f"), "---", "---", "---"]) + r" \\",
    ]
    for B in (5, 8, 16):
        k = f"/by_B/B={B}"
        d1 = ev.reduce("sum", TMS_ART, f"{k}/thermometer_simulated/n_differing_f1_op_per_seed/*", fmt=",")
        d2 = ev.reduce("sum", TMS_ART, f"{k}/thermometer_simulated/n_differing_sens80_op_per_seed/*", fmt=",")
        mm = ev.get(TMS_ART, f"{k}/thermometer_simulated/max_abs_margin_difference_over_seeds", fmt="sci2")
        f1f, f1s = ev.get(TMS_ART, f"{k}/near_threshold/f1/total_flipped", fmt=","), ev.get(TMS_ART, f"{k}/near_threshold/f1/total_stays", fmt=",")
        sf, ss = ev.get(TMS_ART, f"{k}/near_threshold/sens80/total_flipped", fmt=","), ev.get(TMS_ART, f"{k}/near_threshold/sens80/total_stays", fmt=",")
        lines.append(" & ".join([
            f"{B} bins", _pm(ev, TMS_ART, f"{k}/quantized_plain/auroc_test", ".4f"),
            _pm(ev, TMS_ART, f"{k}/quantized_plain/auroc_cost_vs_full_precision", ".4f"),
            _pm(ev, TMS_ART, f"{k}/quantized_plain/f1_f1_op", ".3f"),
            f"{d1} / {d2}", mm, f"{f1f} / {f1s}; {sf} / {ss}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t15_thermometer_exactness", "t15_thermometer_exactness.tex", lines, ev)


def table_s8_locked_chain() -> None:
    ev = Evidence("s8_locked_chain")
    ev.get(TMS_ART, "/source", note=f"REAL ciphertext execution on the locked chain (added arm). {PROXY}. {TM_NOTE}")
    n = ev.get(TMS_ART, "/locked_chain_arm/by_B/B=5/packed_thermometer_locked_chain/n_stays")
    lv = ev.get(TMS_ART, "/locked_chain_levels")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Added arm: the thermometer circuit on the locked {lv}-level chain, seed 42, {n} test stays per bin count on \textsc{{real}} ciphertexts. This arm separates the protocol from "
        r"the shorter chain used in Table~\ref{tab:t14_thermometer_real}. No threshold is applied to these costs; they are reported next to the four-level and interpolation "
        r"rows of that table. Seconds are wall-clock per stay for one label. " + PROXY + r".}",
        r"\label{tab:s8_locked_chain}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Circuit & $B$ & Upload (MB) & Seconds per stay & Max margin error & Agreeing decisions \\",
        r"\midrule",
    ]
    for label, key in (("Thermometer, broadcast", "broadcast_thermometer_locked_chain"), ("Thermometer, packed", "packed_thermometer_locked_chain")):
        for B in (5, 8, 16):
            r = f"/locked_chain_arm/by_B/B={B}/{key}"
            lines.append(" & ".join([
                label if B == 5 else "", str(B), ev.get(TMS_ART, f"{r}/client_upload_mb", fmt=".2f"),
                ev.get(TMS_ART, f"{r}/mean_seconds_per_stay", fmt=".2f") + r" $\pm$ " + ev.get(TMS_ART, f"{r}/sd_seconds_per_stay", fmt=".2f"),
                ev.get(TMS_ART, f"{r}/max_abs_margin_error", fmt="sci2"),
                f"{ev.get(TMS_ART, f'{r}/decisions_agreeing')} of {ev.get(TMS_ART, f'{r}/n_stays')}"]) + r" \\")
        if key.startswith("broadcast"):
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("s8_locked_chain", "s8_locked_chain.tex", lines, ev)


def table_t19_near_threshold() -> None:
    ev = Evidence("t19_near_threshold")
    ev.get(TMS_ART, "/source", note=f"Added analysis: stays within the gate of a threshold, REAL ciphertexts with repeated fresh encryptions; simulated decisions. {PROXY}. {TM_NOTE}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Added analysis: stays whose plaintext margin lies within the correctness gate of a decision threshold, summed over the five seeds, and what happened to them (thermometer circuits, full test split). "
        r"The range is per seed and bin count, over stays within the gate of either threshold. Flipped: changed decision in at least one repetition on \textsc{real} ciphertexts with fresh encryptions; "
        r"with fewer repetitions for four of the five seeds the flip counts are lower bounds. Simulated: decisions of the \textsc{simulated} circuit that differ from the plaintext model. " + PROXY + r".}",
        r"\label{tab:t19_near_threshold}",
        r"\begin{tabular}{rrrrrrrr}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{F1 threshold} & \multicolumn{3}{c}{80\% sensitivity threshold} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
        r"$B$ & Per seed, range & Stays & Flipped & Simulated & Stays & Flipped & Simulated \\",
        r"\midrule",
    ]
    for B in (5, 8, 16):
        k = f"B={B}"
        lo = ev.reduce("min", TMS_ART, f"/by_B/{k}/thermometer_simulated/stays_near_either_threshold_per_seed/*", fmt=",")
        hi = ev.reduce("max", TMS_ART, f"/by_B/{k}/thermometer_simulated/stays_near_either_threshold_per_seed/*", fmt=",")
        cells = [str(B), f"{lo} to {hi}"]
        for thr in ("f1", "sens80"):
            r = f"/by_B/{k}/near_threshold/{thr}"
            cells += [ev.get(TMS_ART, f"{r}/total_stays", fmt=","), ev.get(TMS_ART, f"{r}/total_flipped", fmt=","),
                      ev.reduce("sum", TMS_ART, f"{r}/n_simulated_differing_per_seed/*", fmt=",")]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t19_near_threshold", "t19_near_threshold.tex", lines, ev)


FC_ART = "m1c_formal_counts.json"


def table_t18_derived_costs() -> None:
    ev = Evidence("t18_derived_costs")
    ev.get(FC_ART, "/note", note=f"Added analysis: counts derived by running the real evaluation functions on a depth-tracking mock; saved model files only. {PROXY}.")
    chain4 = ev.get(TMS_ART, "/chain_levels/four_level")
    chain7 = ev.get(TMS_ART, "/chain_levels/locked_seven_level")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Added analysis: costs of the three exact circuits derived from the circuits and set against the measured upload. Levels, ciphertext counts, multiplications and "
        r"rotation calls come from running the evaluation functions of the study on a mock ciphertext that tracks the multiplicative level, over the five saved models of each bin count "
        r"(the ranges are over seeds); nothing is encrypted. A rotation call is one call of the library's rotation, which is realised by one or more key switches that are not counted. "
        rf"The chains that ran have {chain4} levels (thermometer circuits) and {chain7} levels (interpolation). The last column is the measured per-stay upload relative to the packed circuit "
        r"(same-chain run, Table~\ref{tab:t17_samechain}); the packed circuit is the reference. At 16 bins the exact interpolation polynomial could not be constructed in double precision, so its row "
        r"uses placeholder coefficients, which do not change the counts. " + PROXY + r".}",
        r"\label{tab:t18_derived_costs}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Circuit & $B$ & Client ciphertexts & Levels & $\mathrm{ct}\times\mathrm{ct}$ & $\mathrm{ct}\times\mathrm{plain}$ & Rotation calls & Upload relative to packed \\",
        r"\midrule",
    ]
    def rng(lo: str, hi: str) -> str:
        return lo if lo == hi else f"{lo} to {hi}"
    for label, c in (("Thermometer, broadcast", "broadcast"), ("Thermometer, packed", "packed"), ("Exact interpolation", "interpolation")):
        for B in (5, 8, 16):
            k = f"B={B}"
            r = f"/by_B/{k}/seeds/*/{c}"
            d = f"/by_B/{k}/design"
            ctct = rng(ev.reduce("min", FC_ART, f"{r}/ct_ct_multiplications", fmt=","), ev.reduce("max", FC_ART, f"{r}/ct_ct_multiplications", fmt=","))
            ctpl = rng(ev.reduce("min", FC_ART, f"{r}/ct_plain_multiplications", fmt=","), ev.reduce("max", FC_ART, f"{r}/ct_plain_multiplications", fmt=","))
            rot = rng(ev.reduce("min", FC_ART, f"{r}/rotation_calls", fmt=","), ev.reduce("max", FC_ART, f"{r}/rotation_calls", fmt=","))
            if c == "broadcast":
                rel = ev.get(FC_ART, f"/by_B/{k}/measured_upload_ratio_broadcast_to_packed", fmt=".2f")
            elif c == "interpolation":
                rel = ev.get(FC_ART, f"/by_B/{k}/measured_upload_ratio_interpolation_to_packed", fmt=".2f") if B == 5 else "not run"
            else:
                rel = ev.literal("1.00", reason="the packed circuit is the reference of the ratio")
            lines.append(" & ".join([
                label if B == 5 else "", str(B), str(ev.get(FC_ART, f"{d}/client_ciphertexts/{c}")), str(ev.get(FC_ART, f"{d}/levels/{c}")),
                ctct, ctpl, rot, rel]) + r" \\")
        if c != "interpolation":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t18_derived_costs", "t18_derived_costs.tex", lines, ev)


SC_ART = "m1c_thermo_samechain_summary.json"


def table_t17_samechain() -> None:
    ev = Evidence("t17_samechain")
    ev.get(SC_ART, "/prespec", note=f"Added study (same-chain comparison, five seeds), REAL ciphertexts on the locked chain. {PROXY}.")
    n_seeds = ev.reduce("len", SC_ART, "/seeds/*")
    per_seed = ev.get(SC_ART, "/n_stays_per_seed_and_B")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Added study: the three exact circuits on the same locked seven-level chain, in one session per seed, on \textsc{{real}} ciphertexts. {per_seed} stays per seed and bin count over {n_seeds} seeds "
        r"(the frozen stratified sample of Table~\ref{tab:t14_thermometer_real}; 50 plaintext-positive and 50 plaintext-negative at the F1-maximizing threshold). Seconds per stay are mean $\pm$ SD over seeds of the per-seed mean, "
        r"the circuits alternating in a rotating order within each stay. The sample was drawn around the F1 threshold, so agreement there is expected by construction; the 80\%-sensitivity threshold was not used to draw it. "
        r"Margin gaps are in Table~\ref{tab:s11_margin_gaps}. The one-seed arm of Table~\ref{tab:s8_locked_chain} is unchanged. " + PROXY + r".}",
        r"\label{tab:t17_samechain}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Circuit & $B$ & Upload (MB) & Seconds per stay & Max margin error & Agree, F1 point & Agree, 80\% point \\",
        r"\midrule",
    ]
    names = (("Thermometer, broadcast", "broadcast"), ("Thermometer, packed", "packed"), ("Exact interpolation", "interpolation"))
    first = True
    for label, c in names:
        if not first:
            lines.append(r"\midrule")
        first = False
        for B in (5, 8, 16):
            if c == "interpolation" and B != 5:
                continue
            r = f"/by_B/B={B}/circuits/{c}"
            n = ev.get(SC_ART, f"{r}/n_stays_total", fmt=",")
            lines.append(" & ".join([
                label if (B == 5) else "", str(B), ev.get(SC_ART, f"{r}/client_upload_mb", fmt=".2f"),
                _pm(ev, SC_ART, f"{r}/mean_seconds", ".2f"),
                ev.get(SC_ART, f"{r}/max_abs_margin_error_over_seeds", fmt="sci2"),
                f"{ev.get(SC_ART, f'{r}/decisions_agreeing_f1_total', fmt=',')} of {n}",
                f"{ev.get(SC_ART, f'{r}/decisions_agreeing_sens80_total', fmt=',')} of {n}"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("t17_samechain", "t17_samechain.tex", lines, ev)


def table_s11_margin_gaps() -> None:
    ev = Evidence("s11_margin_gaps")
    ev.get(SC_ART, "/prespec", note=f"Added analysis (margin gaps), plaintext margins against the two thresholds. {PROXY}.")
    gate = ev.literal("5\\times10^{-4}", reason="correctness gate of the signed plan, declared constant")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Added analysis: distance between an attained plaintext margin and a decision threshold, smallest over the five seeds, and the number of test stays lying within the correctness gate (${gate}$) of a threshold. "
        r"The \emph{sample} columns are over the sampled stays of each seed (the stays run on ciphertexts); the \emph{full split} columns are over the whole held-out test split. "
        r"The count is the range over seeds of the test stays that could change decision under an error of at most the gate; it bounds, and does not count, the flips. " + PROXY + r".}",
        r"\label{tab:s11_margin_gaps}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Threshold & $B$ & Smallest gap, sample & Smallest gap, full split & Stays within gate (min--max over seeds) \\",
        r"\midrule",
    ]
    for label, key in (("F1 point", "f1"), ("80\% sensitivity", "sens80")):
        for B in (5, 8, 16):
            r = f"/by_B/B={B}/margin_gaps/*/{key}"
            lo = ev.reduce("min", SC_ART, f"{r}/n_full_split_within_gate", fmt=",")
            hi = ev.reduce("max", SC_ART, f"{r}/n_full_split_within_gate", fmt=",")
            lines.append(" & ".join([
                label if B == 5 else "", str(B),
                ev.reduce("min", SC_ART, f"{r}/min_gap_sample_100", fmt="sci2"),
                ev.reduce("min", SC_ART, f"{r}/min_gap_full_test_split", fmt="sci2"),
                f"{lo}--{hi}"]) + r" \\")
        if key == "f1":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("s11_margin_gaps", "s11_margin_gaps.tex", lines, ev)


def table_s9_keys() -> None:
    ev = Evidence("s9_keys")
    ev.get(TMS_ART, "/source", note=f"One-off key material of the REAL runs. {PROXY}. {TM_NOTE}")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{One-off key material, serialized size in MB (secret key excluded), for the two coefficient chains used on \textsc{real} ciphertexts. "
        r"The Galois keys are TenSEAL's default set of power-of-two rotations; no key minimization was attempted, so these are upper bounds for the "
        r"rotations actually used. The per-stay upload in Table~\ref{tab:t14_thermometer_real} excludes these. " + PROXY + r".}",
        r"\label{tab:s9_keys}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Chain & Public key and parameters & Relinearization & Galois & Total \\",
        r"\midrule",
    ]
    for label, key in (("Four-level", "n16384_four_level"), ("Locked, seven-level", "n16384_locked_seven_level")):
        r = f"/key_bytes_one_off/{key}"
        lines.append(" & ".join([label] + [ev.get(TMS_ART, f"{r}/{f}_mb", fmt=".1f")
                                           for f in ("public_key_and_params", "relin_keys", "galois_keys", "total")]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    finish("s9_keys", "s9_keys.tex", lines, ev)


def table_s10_pilot_checks() -> None:
    ev = Evidence("s10_pilot_checks")
    art = "m1c_thermo_pilot.json"
    ev.get(art, "/protocol", note=f"Validity pilot of the structure- and threshold-hiding protocol (added study). {PROXY}.")
    r = "/results/B=5/broadcast_thermometer_ring8192"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Two checks from the validity pilot of the structure- and threshold-hiding protocol (broadcast variant, 5 bins; \textsc{real} ciphertexts). "
        r"The smaller ring degree failed the margin gate, and TenSEAL's symmetric encryption gave no reduction in ciphertext size. " + PROXY + r".}",
        r"\label{tab:s10_pilot_checks}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Ring degree & Stays & Largest margin error & Gate & Decisions agreeing & One-off keys (bytes) \\",
        r"\midrule",
        " & ".join([
            ev.literal("8,192", reason="ring degree tested in the pilot, a design parameter and not a result"), str(ev.get(art, f"{r}/n_stays")), ev.get(art, f"{r}/max_abs_margin_error", fmt="sci2"), ev.get(art, f"{r}/gate", fmt="sci2"),
            f"{ev.get(art, f'{r}/decisions_agreeing')} of {ev.get(art, f'{r}/n_stays')}",
            ev.get(art, "/key_bytes_one_off/n8192/total", fmt=",")]) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        r"\parbox{\linewidth}{\footnotesize Serialization check: " + tex_escape(ev.get(art, "/seeded_serialization_check")) + r".}",
        r"\end{table}",
    ]
    finish("s10_pilot_checks", "s10_pilot_checks.tex", lines, ev)


TREE_SYSTEMS_JSON = os.path.join(_HERE, "..", "paper_assets", "tree_systems.json")


def table_t16_tree_systems() -> None:
    ev = Evidence("t16_tree_systems")
    with open(TREE_SYSTEMS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    mark_cols = ("ensemble", "ckks", "noninteractive", "hidden", "bootstrapping")
    symbol = {"yes": r"$\checkmark$", "no": r"$\times$", "nr": "--"}
    letters = "abcdefghij"
    notes: list = []  # footnote texts in order of appearance

    def mark_cells(p: dict, external: bool) -> list:
        cells = []
        for c in mark_cols:
            sym = symbol[p["marks"][c]]
            note = p.get("mark_notes", {}).get(c)
            if note:
                if external:
                    ev.external(note, citation=p["citation"], locus=p["locus"], note=f"footnote on column '{c}' as read from: {p['locus']}")
                else:
                    ev.literal(note, reason=p["reason"])
                if note not in notes:
                    notes.append(note)
                sym += rf"$^{{{letters[notes.index(note)]}}}$"
            cells.append(sym)
        return cells

    rows = []
    for p in data["papers"]:
        short = ev.literal(p["short_name"], reason="citation display label (author(s) + year), not a measured value")
        # the marks are read off the row's text cells (model, scheme, interaction, hidden), which stay in the provenance record
        for c in ("model", "scheme", "interaction", "hidden"):
            ev.external(p[c], citation=p["citation"], locus=p["locus"], note=f"cell '{c}' as read from: {p['locus']}")
        cells = [tex_escape(short) + rf"~\cite{{{p['citation']}}}"]
        cells += mark_cells(p, True)
        if p.get("log2_ring_degree"):
            cells.append(rf"$2^{{{ev.external(str(p['log2_ring_degree']), citation=p['citation'], locus=p['locus'], note='log2 of the ring degree, Table 4')}}}$")
        else:
            cells.append("--")
        cells.append(tex_escape(ev.external(p["comparison"], citation=p["citation"], locus=p["locus"], note=f"cell 'comparison' as read from: {p['locus']}")))
        rows.append(cells)
    mine = []
    for p in data["this_work"]:
        cells = [r"\textbf{" + tex_escape(ev.literal(p["short_name"], reason=p["reason"])) + "}"]
        cells += mark_cells(p, False)
        n_ring = int(str(ev.get("m1_ckks_modulus_chain.json", "/N_poly_modulus_degree")).replace(",", ""))
        exp_ring = ev.literal(str(int(round(math.log2(n_ring)))), reason="log2 of the ring degree recorded just above, shown as a power of two like the other rows")
        cells.append(f"$2^{{{exp_ring}}}$")
        cells.append(tex_escape(ev.literal(p["comparison"], reason=p["reason"])))
        mine.append(cells)
    foot = "; ".join(rf"$^{{{letters[i]}}}$~{tex_escape(n)}" for i, n in enumerate(notes))
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Private tree-evaluation systems compared by what each paper states. $\checkmark$: stated by the paper; $\times$: stated not; --: not reported in the part read, which does not mean "
        r"that the paper lacks it (NR in the last column has the same meaning). Each mark is taken from the part of the cited paper named in the provenance record (an abstract or the full text). Costs are not tabulated: the systems differ "
        r"in model, hardware, precision and security level, and none was measured on the task of this study. The rows of this work come from the " + PROXY + r" task; the other rows are what their papers state.}",
        r"\label{tab:t16_tree_systems}",
        r"\footnotesize",
        r"\begin{tabular}{>{\raggedright\arraybackslash}p{3.4cm}cccccc>{\raggedright\arraybackslash}p{4.4cm}}",
        r"\toprule",
        r"System & Gradient-boosted ensemble & CKKS & Non-interactive & Model hidden from client & Boot\-strapping & Ring degree & Comparison of an input with a threshold \\",
        r"\midrule",
    ]
    for cells in rows:
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for cells in mine:
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
              r"\parbox{\linewidth}{\footnotesize Notes: " + foot + r".}", r"\end{table}"]
    finish("t16_tree_systems", "t16_tree_systems.tex", lines, ev)


def main() -> int:
    table_t1_dataset_task()
    table_t2_plaintext_baseline()
    table_t3_ckks_circuit_config()
    table_t4_degree_grid()
    table_t5_real_vs_simulated()
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", E2_ARTIFACT)):
        table_t10_real_full_split()
    table_t6_scaling()
    table_t7_error_decomposition()
    table_t8_security_resource()
    table_t9_related_work()
    table_s1_k_selection()
    table_s2_tfhe_calibration()
    table_s3_gate_m1a()
    if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m1_decision_agreement.json")):
        table_s4_decision_agreement()
    _res = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
    if os.path.exists(os.path.join(_res, "m1_interleaved_latency_summary.json")):
        table_s5_interleaved_latency()
    if os.path.exists(os.path.join(_res, "m1_openfhe_ps_benchmark.json")):
        table_s6_ps_benchmark()
    if os.path.exists(os.path.join(_res, "m1c_final_summary.json")):
        table_t11_continuous_fidelity()
        table_t12_continuous_utility()
        table_s7_pilot_trajectory()
    if os.path.exists(os.path.join(_res, "m1c_final_real.json")):
        table_t13_continuous_real()
    if os.path.exists(os.path.join(_res, "m1c_thermo_final_summary.json")):
        table_t14_thermometer_real()
        table_t15_thermometer_exactness()
        table_s8_locked_chain()
        table_s9_keys()
    if os.path.exists(os.path.join(_res, "m1c_thermo_samechain_summary.json")):
        table_t17_samechain()
        table_s11_margin_gaps()
    if os.path.exists(os.path.join(_res, "m1c_formal_counts.json")):
        table_t18_derived_costs()
        table_t19_near_threshold()
    if os.path.exists(os.path.join(_res, "m1c_thermo_pilot.json")):
        table_s10_pilot_checks()
    if os.path.exists(os.path.join(_res, "..", "paper_assets", "tree_systems.json")):
        table_t16_tree_systems()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
