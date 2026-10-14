"""
M1 -- Named macro registry for numbers quoted in manuscript PROSE (not inside a generated table,
which already carries its own numbers) or in the F1/F2 TikZ diagrams. See
src/paper_data.py:emit_macro_file for the design rationale (explicit, stable names -- never
positional).

Only headline/repeated quantities that a manuscript SENTENCE states directly are registered here.
A number that only ever appears inside a table cell does not need a macro: it is already traced by
that table's own sidecar, and restating it in prose as well as in the table is redundant, not
extra-safe. The manuscript's own style is to reference the table/figure (\\ref{tab:t4_degree_grid})
for exact per-cell values and use a macro only for the number a sentence actually asserts.

Output: paper_assets/tables/m1_numbers.tex + paper_assets/provenance/m1_numbers.json
Usage: python generate_paper_macros.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_data import emit_macro_file  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "..", "paper_assets", "tables", "m1_numbers.tex")

# (macro_suffix, kind, artifact, pointer, fmt, note)
# kind = "get" (one value at one pointer) or a reduce() method name ("max"/"min"/"mean"/"len").
MACROS: list[tuple[str, str, str, str, str | None, str]] = [
    # ---- Dataset / cohort (guide Sec 10) ------------------------------------------------
    ("CohortStays", "get", "m1_cohort_flow.json", "/final_cohort_size", ",", "Proxy cohort (MIMIC-IV-ED-derived)."),
    ("FeatureDim", "get", "m1_plaintext_baseline.json", "/feature_dim", None, ""),
    ("NLabels", "get", "m1_label_space.json", "/top_k", None, ""),
    ("TrainStays", "get", "m1_plaintext_baseline.json", "/split/train", ",", ""),
    ("ValStays", "get", "m1_plaintext_baseline.json", "/split/val", ",", ""),
    ("TestStays", "get", "m1_plaintext_baseline.json", "/split/test", ",", ""),
    ("AtcCoverage", "get", "m1_atc_crosswalk_coverage.json", "/row_weighted_coverage_rate", ".1%", ""),

    # ---- Plaintext baseline (guide Sec 11) ----------------------------------------------
    ("MacroFOneMean", "get", "m1_plaintext_baseline.json", "/macro_f1_tuned/mean", ".4f", ""),
    ("MacroFOneSD", "get", "m1_plaintext_baseline.json", "/macro_f1_tuned/std", ".4f", ""),
    ("MeanAuroc", "get", "m1_plaintext_baseline.json", "/mean_auroc/mean", ".4f", ""),
    ("MeanAuprc", "get", "m1_plaintext_baseline.json", "/mean_auprc/mean", ".4f", ""),
    ("NTrees", "get", "m1_plaintext_baseline.json", "/n_estimators", None, ""),
    ("MaxDepth", "get", "m1_plaintext_baseline.json", "/max_depth", None, ""),

    # ---- CKKS configuration (guide Sec 12) ----------------------------------------------
    ("RingDegree", "get", "m1_ckks_modulus_chain.json", "/N_poly_modulus_degree", ",", ""),
    ("ScaleBits", "get", "m1_ckks_modulus_chain.json", "/scale_bits", None, ""),
    ("TotalModulusBits", "get", "m1_ckks_modulus_chain.json", "/total_modulus_bits_sum_of_bit_lengths", None, ""),
    ("SecurityStandardCeilingBits", "get", "m1_security_level_verification.json", "/by_security_level/TC128/standard_max_bits_for_N", None, ""),
    ("GammaValue", "get", "m1_encrypted_multiseed.json", "/coefficients/gamma", ".4f", ""),
    ("AnchorPercentile", "get", "m1_encrypted_multiseed.json", "/coefficients/gamma_anchor_percentile", None, ""),
    ("ApproxIntervalHalfWidth", "get", "m1_circuit_params.json", "/approximation_interval/a_half_width", ".4f", ""),
    # Max LEAVES (occupied slots), not max internal SPLITS -- these are different quantities
    # (2,751 vs 2,701); conflating them was the "up to 2,751 slots per leaf" error already
    # corrected for the supervisor in docs/guide_verification_note.md. Same reduce() as
    # table_t3_ckks_circuit_config in generate_latex_tables.py.
    ("MaxLeaves", "max", "m1_circuit_params.json", "/split_census/per_model/*/n_leaves", ",",
     "max over 25 frozen models; this is LEAVES (occupied slots), not internal splits"),
    ("SlotsAtN", "get", "m1_circuit_params.json", "/depth_budget/k=5/slots_at_that_degree", ",", ""),
    ("TreeDepthLevels", "get", "m1_circuit_params.json", "/tree_depth", None, ""),
    ("NBinaryFeatures", "get", "m1_circuit_params.json", "/n_binary_features", None, ""),
    ("NContinuousFeatures", "get", "m1_circuit_params.json", "/n_continuous_features", None, ""),
    ("NTotalSplits", "get", "m1_circuit_params.json", "/split_census/total_internal_splits", ",",
     "internal splits summed over the 25 frozen models"),
    ("NBinarySplits", "get", "m1_circuit_params.json", "/split_census/total_binary_splits", ",",
     "splits on binary features; evaluated exactly (degree-1 polynomial) at every k"),
    ("NContinuousSplits", "get", "m1_circuit_params.json", "/split_census/total_continuous_splits", ",",
     "splits on continuous features; the only splits that use the Chebyshev approximation"),
    ("ContinuousSplitFraction", "get", "m1_circuit_params.json", "/split_census/fraction_continuous_splits", ".3f", ""),
    # added analysis S2.2 (docs/step2_prespec_m1.md section A): shares of pooled |margin| near zero, by split type
    ("ContMarginShareBelowSmall", "get", "m1_margin_distribution.json", "/continuous_feature_splits/share_abs_lt_0.05", ".3f",
     "share of continuous-feature-split margins with |z| < 0.05"),
    ("ContMarginShareBelowMid", "get", "m1_margin_distribution.json", "/continuous_feature_splits/share_abs_lt_0.10", ".3f",
     "share with |z| < 0.10"),
    ("ContMarginShareBelowLarge", "get", "m1_margin_distribution.json", "/continuous_feature_splits/share_abs_lt_0.20", ".3f",
     "share with |z| < 0.20"),
    ("BinMarginShareBelowSmall", "get", "m1_margin_distribution.json", "/binary_feature_splits/share_abs_lt_0.05", ".3f",
     "share of binary-feature-split margins with |z| < 0.05"),
    ("BinMarginShareBelowLarge", "get", "m1_margin_distribution.json", "/binary_feature_splits/share_abs_lt_0.20", ".3f",
     "share with |z| < 0.20 (equal to the share below 0.05: no binary margin lies between 0.05 and 0.20)"),
    # added analysis S2.1 (docs/step2_prespec_m1.md section B): decision disagreement, plaintext vs simulated CKKS
    ("DecPerThousandKOne", "get", "m1_decision_agreement.json", "/by_k/k=1/stays_per_1000_mean", ".1f", "stays per 1,000 whose decision differs; mean over labels and seeds"),
    ("DecMaxCellKOne", "get", "m1_decision_agreement.json", "/by_k/k=1/max_cell_rate", ".3f", "largest label-seed cell, fraction of test stays"),
    ("DecPerThousandKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/stays_per_1000_mean", ".1f", "stays per 1,000 whose decision differs; mean over labels and seeds"),
    ("DecMaxCellKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/max_cell_rate", ".3f", "largest label-seed cell, fraction of test stays"),
    ("DecPerThousandKFive", "get", "m1_decision_agreement.json", "/by_k/k=5/stays_per_1000_mean", ".1f", "stays per 1,000 whose decision differs; mean over labels and seeds"),
    ("DecMaxCellKFive", "get", "m1_decision_agreement.json", "/by_k/k=5/max_cell_rate", ".3f", "largest label-seed cell, fraction of test stays"),
    ("DecPerThousandKSeven", "get", "m1_decision_agreement.json", "/by_k/k=7/stays_per_1000_mean", ".1f", "stays per 1,000 whose decision differs; mean over labels and seeds"),
    ("DecMaxCellKSeven", "get", "m1_decision_agreement.json", "/by_k/k=7/max_cell_rate", ".3f", "largest label-seed cell, fraction of test stays"),
    ("DecPerThousandKNine", "get", "m1_decision_agreement.json", "/by_k/k=9/stays_per_1000_mean", ".1f", "stays per 1,000 whose decision differs; mean over labels and seeds"),
    ("DecMaxCellKNine", "get", "m1_decision_agreement.json", "/by_k/k=9/max_cell_rate", ".3f", "largest label-seed cell, fraction of test stays"),
    ("DecSdKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/sd_across_seeds_of_label_mean", ".4f", "SD across seeds of the label-mean rate"),
    ("DecCiLoKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/cluster_bootstrap_ci95_of_label_seed_mean/0", ".4f", "patient-cluster bootstrap 95% lower"),
    ("DecCiHiKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/cluster_bootstrap_ci95_of_label_seed_mean/1", ".4f", "patient-cluster bootstrap 95% upper"),
    ("DecRateKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/mean_over_labels_and_seeds", ".4f", "mean disagreement rate"),
    ("DecSdKNine", "get", "m1_decision_agreement.json", "/by_k/k=9/sd_across_seeds_of_label_mean", ".4f", "SD across seeds of the label-mean rate"),
    ("DecCiLoKNine", "get", "m1_decision_agreement.json", "/by_k/k=9/cluster_bootstrap_ci95_of_label_seed_mean/0", ".4f", "patient-cluster bootstrap 95% lower"),
    ("DecCiHiKNine", "get", "m1_decision_agreement.json", "/by_k/k=9/cluster_bootstrap_ci95_of_label_seed_mean/1", ".4f", "patient-cluster bootstrap 95% upper"),
    ("DecRateKNine", "get", "m1_decision_agreement.json", "/by_k/k=9/mean_over_labels_and_seeds", ".4f", "mean disagreement rate"),
    ("DecProbErrMeanKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/prob_abs_error_mean_over_labels_and_seeds", ".3f", "mean |p_plain - p_ckks|"),
    ("DecProbErrMaxKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/prob_abs_error_max_over_all_cells", ".2f", "largest single-stay |p_plain - p_ckks| in any cell"),
    ("DecAurocAbsMaxKThree", "get", "m1_decision_agreement.json", "/by_k/k=3/auroc_diff_abs_max_over_all_cells", ".3f", "largest |AUROC_plain - AUROC_ckks| over label-seed cells"),
    # added analyses S2.3 and S2.4 (docs/step2_prespec_m1.md sections C and D)
    ("InterleavedRatioKThree", "get", "m1_interleaved_latency_summary.json", "/by_k/k=3/folding_over_original_ratio_mean", ".2f", "folding / original latency, mean of the two interleaved rounds"),
    ("InterleavedRatioKFive", "get", "m1_interleaved_latency_summary.json", "/by_k/k=5/folding_over_original_ratio_mean", ".2f", "folding / original latency, mean of the two interleaved rounds"),
    ("InterleavedRatioKSeven", "get", "m1_interleaved_latency_summary.json", "/by_k/k=7/folding_over_original_ratio_mean", ".2f", "folding / original latency, mean of the two interleaved rounds"),
    ("FiveLabelWallS", "get", "m1_five_label_cost.json", "/five_label_total_per_patient/wall_clock_s/mean", ".2f", "encrypt + server + decrypt, five labels, one patient"),
    ("FiveLabelWallSD", "get", "m1_five_label_cost.json", "/five_label_total_per_patient/wall_clock_s/sd", ".2f", "SD over the 20 timed patients"),
    ("FiveLabelUploadBytes", "get", "m1_five_label_cost.json", "/five_label_total_per_patient/upload_bytes_sum_over_labels", ",", "bytes uploaded per patient, five labels"),
    ("FiveLabelPublicContextBytes", "get", "m1_five_label_cost.json", "/context_bytes/public_context_incl_evaluation_keys", ",", "serialized public context incl. evaluation keys"),
    ("PsLevelsScalarKThree", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=3/levels_consumed_scalar", None, "levels consumed by OpenFHE PS Chebyshev evaluation"),
    ("PsLevelsAffineKThree", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=3/levels_consumed_with_per_slot_affine", None, "levels with the per-slot affine step"),
    ("PsLevelsScalarKFive", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=5/levels_consumed_scalar", None, "levels consumed by OpenFHE PS Chebyshev evaluation"),
    ("PsLevelsAffineKFive", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=5/levels_consumed_with_per_slot_affine", None, "levels with the per-slot affine step"),
    ("PsLevelsScalarKSeven", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=7/levels_consumed_scalar", None, "levels consumed by OpenFHE PS Chebyshev evaluation"),
    ("PsLevelsAffineKSeven", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=7/levels_consumed_with_per_slot_affine", None, "levels with the per-slot affine step"),
    ("PsLevelsScalarKNine", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=9/levels_consumed_scalar", None, "levels consumed by OpenFHE PS Chebyshev evaluation"),
    ("PsLevelsAffineKNine", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=9/levels_consumed_with_per_slot_affine", None, "levels with the per-slot affine step"),
    ("PsTimeScalarKThree", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=3/scalar_polynomial/mean_s", ".3f", "OpenFHE PS time per ciphertext"),
    ("SealFoldScalarTimeKThree", "get", "m1_seal_scalar_poly_benchmark.json", "/by_k/k=3/folding/mean_s", ".3f", "SEAL folding evaluator time per ciphertext"),
    ("PsTimeScalarKNine", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=9/scalar_polynomial/mean_s", ".3f", "OpenFHE PS time per ciphertext"),
    ("SealFoldScalarTimeKNine", "get", "m1_seal_scalar_poly_benchmark.json", "/by_k/k=9/folding/mean_s", ".3f", "SEAL folding evaluator time per ciphertext"),
    ("SealOrigScalarTimeKThree", "get", "m1_seal_scalar_poly_benchmark.json", "/by_k/k=3/original/mean_s", ".3f", "SEAL original evaluator time per ciphertext"),
    ("SealOrigScalarTimeKSeven", "get", "m1_seal_scalar_poly_benchmark.json", "/by_k/k=7/original/mean_s", ".3f", "SEAL original evaluator time per ciphertext"),
    ("PsTimeScalarKSeven", "get", "m1_openfhe_ps_benchmark.json", "/by_k/k=7/scalar_polynomial/mean_s", ".3f", "OpenFHE PS time per ciphertext"),
    ("MarginMedianAbs", "get", "m1_circuit_params.json", "/approximation_interval/abs_diff_quantiles/50", ".2f",
     "median of the pooled absolute normalized margins of the CONTINUOUS-feature splits only (training split)"),
    ("MarginUpperQuartileAbs", "get", "m1_circuit_params.json", "/approximation_interval/abs_diff_quantiles/75", ".2f",
     "75th percentile of the pooled absolute normalized margins of the CONTINUOUS-feature splits only (training split)"),
    ("ChainLevelsAvailable", "get", "m1_depth_probe.json", "/per_degree/k=3/levels_available_on_locked_chain", None, ""),

    # ---- Degree sweep / utility (guide Sec 14) ------------------------------------------
    ("UtilityTolerance", "get", "m1_ckks_sweep_simulated.json", "/utility_tolerance", None, "SIMULATED."),
    ("DeltaFOneDegThree", "get", "m1_ckks_sweep_simulated.json", "/k_sweep/k=3/delta_f1_mean_over_seeds", ".4f", "SIMULATED."),
    ("DeltaFOneDegFive", "get", "m1_ckks_sweep_simulated.json", "/k_sweep/k=5/delta_f1_mean_over_seeds", ".4f", "SIMULATED."),
    ("DeltaFOneDegSeven", "get", "m1_ckks_sweep_simulated.json", "/k_sweep/k=7/delta_f1_mean_over_seeds", ".4f", "SIMULATED."),
    ("DeltaFOneDegNine", "get", "m1_ckks_sweep_simulated.json", "/k_sweep/k=9/delta_f1_mean_over_seeds", ".4f",
     "SIMULATED; the k=9 utility result is simulation only (the original evaluator needs 8 levels, 7 are available)."),
    ("DeltaFOneDegOne", "get", "m1_ckks_sweep_k1_simulated.json", "/delta_f1_mean_over_seeds", ".4f",
     "SIMULATED; k=1 is an additional diagnostic arm outside the locked degree family {3,5,7,9}."),
    ("DepthDegOne", "get", "m1_depth_budget_k1.json", "/code_derived/total_multiplicative_depth", None, ""),
    ("DepthDegThree", "get", "m1_depth_probe.json", "/per_degree/k=3/code_derived/total_multiplicative_depth", None, ""),
    ("DepthDegFive", "get", "m1_depth_probe.json", "/per_degree/k=5/code_derived/total_multiplicative_depth", None, ""),
    ("DepthDegSeven", "get", "m1_depth_probe.json", "/per_degree/k=7/code_derived/total_multiplicative_depth", None, ""),
    ("DepthDegNine", "get", "m1_depth_probe.json", "/per_degree/k=9/code_derived/total_multiplicative_depth", None, ""),
    ("CoeffBitsDegNine", "get", "m1_depth_probe.json", "/per_degree/k=9/coeff_modulus_bits_needed_for_this_depth", None, ""),

    # ---- Real vs simulated validation (guide Sec 15) ------------------------------------
    ("RealNPerSeed", "get", "m1_encrypted_multiseed.json", "/n_per_seed", None, ""),
    ("RealNTotal", "sum", "m1_encrypted_multiseed.json", "/per_cell/*/n", None, "REAL ciphertext evaluations over all seeds and executed degrees."),
    ("RealLatencyDegThree", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=3/latency_wall_clock_s_mean/mean", ".3f", "REAL ciphertext execution."),
    ("RealLatencyDegThreeSD", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=3/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("RealLatencyDegFive", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=5/latency_wall_clock_s_mean/mean", ".3f", "REAL ciphertext execution."),
    ("RealLatencyDegFiveSD", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=5/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("RealLatencyDegSeven", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=7/latency_wall_clock_s_mean/mean", ".3f", "REAL ciphertext execution."),
    ("RealLatencyDegSevenSD", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=7/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("AgreementDegThree", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=3/agreement_real_vs_simulated/mean", ".4f", "REAL ciphertext execution."),
    ("AgreementDegFive", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=5/agreement_real_vs_simulated/mean", ".4f", "REAL ciphertext execution."),
    ("AgreementDegSeven", "get", "m1_encrypted_multiseed.json", "/pooled_across_seeds/k=7/agreement_real_vs_simulated/mean", ".4f", "REAL ciphertext execution."),

    ("GammaTieTolerance", "get", "m1_gamma_anchor_selection.json", "/tie_tolerance", ".3f", "Validation-split anchor selection."),
    ("LargestLabelCellDFOne", "get", "m1_per_label_selection.json", "/largest_cell/abs_delta_f1", ".4f", "SIMULATED; largest label-seed-degree cell."),
    ("LargestLabelCellFrac", "get", "m1_per_label_selection.json", "/largest_cell/fraction_of_tolerance", ".0%", "SIMULATED."),
    ("PerLabelCells", "get", "m1_per_label_selection.json", "/n_label_seed_k_cells", None, "SIMULATED."),
    ("OrigMaxErrAll", "max", "m1_encrypted_multiseed.json", "/per_cell/*/max_abs_error_real_vs_simulated", "sci2", "REAL; original evaluator, largest single-cell error."),
    ("FoldMaxErrAll", "max", "m1_encrypted_multiseed_folded.json", "/per_cell/*/max_abs_error_real_vs_simulated", "sci2", "REAL; folding evaluator, largest single-cell error."),

    # ---- Revision E1: coefficient-folding evaluator (docs/revision_prespec_m1.md) --------
    ("FoldDepthDegThree", "get", "m1_folded_depth_accounting.json", "/per_degree/k=3/total_depth_folded", None, "Code-derived."),
    ("FoldDepthDegFive", "get", "m1_folded_depth_accounting.json", "/per_degree/k=5/total_depth_folded", None, ""),
    ("FoldDepthDegSeven", "get", "m1_folded_depth_accounting.json", "/per_degree/k=7/total_depth_folded", None, ""),
    ("FoldDepthDegNine", "get", "m1_folded_depth_accounting.json", "/per_degree/k=9/total_depth_folded", None, ""),
    ("FoldPolyDepthDegNine", "get", "m1_folded_depth_accounting.json", "/per_degree/k=9/polynomial_depth_folded", None, ""),
    ("OrigPolyDepthDegNine", "get", "m1_folded_depth_accounting.json", "/per_degree/k=9/polynomial_depth_original", None, ""),
    ("OpsCtCtOrigDegNine", "get", "m1_folded_depth_accounting.json", "/per_degree/k=9/polynomial_operations_original/ct_x_ct", None, "Code-derived, per polynomial."),
    ("OpsCtCtFoldDegNine", "get", "m1_folded_depth_accounting.json", "/per_degree/k=9/polynomial_operations_folded/ct_x_ct", None, ""),
    ("OpsCtCtOrigDegThree", "get", "m1_folded_depth_accounting.json", "/per_degree/k=3/polynomial_operations_original/ct_x_ct", None, ""),
    ("OpsCtCtFoldDegThree", "get", "m1_folded_depth_accounting.json", "/per_degree/k=3/polynomial_operations_folded/ct_x_ct", None, ""),
    ("FoldRealNTotal", "sum", "m1_encrypted_multiseed_folded.json", "/per_cell/*/n", None, "REAL ciphertext evaluations, folding evaluator."),
    ("FoldRealNCompleted", "sum", "m1_encrypted_multiseed_folded.json", "/per_cell/*/n_completed", None, "REAL."),
    ("FoldRealNDegNine", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/n_patient_evaluations", None, "REAL."),
    ("FoldAgreeCountDegNine", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/n_sign_agree_real_vs_simulated", None, "REAL."),
    ("FoldNmaeDegNine", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/normalized_mae_real_vs_simulated/mean", "sci2", "REAL."),
    ("FoldNmaeMaxDegNine", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/normalized_mae_real_vs_simulated/max", "sci2", "REAL."),
    ("FoldMaxErrDegNine", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/max_abs_error_real_vs_simulated/max", "sci2", "REAL."),
    ("FoldNmaeMinAll", "min", "m1_encrypted_multiseed_folded.json", "/per_cell/*/normalized_mae_real_vs_simulated", "sci2", "REAL; smallest cell."),
    ("FoldNmaeMaxAll", "max", "m1_encrypted_multiseed_folded.json", "/per_cell/*/normalized_mae_real_vs_simulated", "sci2", "REAL; largest cell."),
    ("OrigNmaeRangeMin", "get", "m1_encrypted_multiseed_folded.json", "/k9_executable_criterion/original_evaluator_nmae_range_k357/min", "sci2", "Original evaluator, 15 cells."),
    ("OrigNmaeRangeMax", "get", "m1_encrypted_multiseed_folded.json", "/k9_executable_criterion/original_evaluator_nmae_range_k357/max", "sci2", "Original evaluator, 15 cells."),
    ("FoldLatencyDegThree", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=3/latency_wall_clock_s_mean/mean", ".3f", "REAL ciphertext execution, folding evaluator."),
    ("FoldLatencyDegThreeSD", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=3/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("FoldLatencyDegFive", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=5/latency_wall_clock_s_mean/mean", ".3f", "REAL."),
    ("FoldLatencyDegFiveSD", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=5/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("FoldLatencyDegSeven", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=7/latency_wall_clock_s_mean/mean", ".3f", "REAL."),
    ("FoldLatencyDegSevenSD", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=7/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("FoldLatencyDegNine", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/latency_wall_clock_s_mean/mean", ".3f", "REAL."),
    ("FoldLatencyDegNineSD", "get", "m1_encrypted_multiseed_folded.json", "/pooled_across_seeds/k=9/latency_wall_clock_s_mean/sd_across_seeds", ".3f", ""),
    ("FoldPolyEvalDegNine", "get", "m1_latency_breakdown_folded.json", "/by_k/k=9/timing_per_patient/poly_eval_s/mean_s", ".3f", "REAL; single machine, seed 42, folding evaluator."),
    ("FoldPolyEvalDegSeven", "get", "m1_latency_breakdown_folded.json", "/by_k/k=7/timing_per_patient/poly_eval_s/mean_s", ".3f", ""),
    ("FoldBreakdownWallDegNine", "get", "m1_latency_breakdown_folded.json", "/by_k/k=9/timing_per_patient/wall_clock_total_s/mean_s", ".3f", ""),
    ("FoldInputCiphertextBytesDegNine", "get", "m1_latency_breakdown_folded.json", "/by_k/k=9/ciphertext_bytes/input_total_per_patient", ",", ""),

    # ---- Latency and resource breakdown (guide Sec 16); one-machine, single-seed (42) arm --
    # separate from the 5-seed m1_encrypted_multiseed.json arm above -- note the distinction in
    # any sentence that cites both.
    ("PolyEvalDegThree", "get", "m1_latency_breakdown.json", "/by_k/k=3/timing_per_patient/poly_eval_s/mean_s", ".3f", "REAL ciphertext execution; single machine, seed 42."),
    ("PolyEvalDegFive", "get", "m1_latency_breakdown.json", "/by_k/k=5/timing_per_patient/poly_eval_s/mean_s", ".3f", ""),
    ("PolyEvalDegSeven", "get", "m1_latency_breakdown.json", "/by_k/k=7/timing_per_patient/poly_eval_s/mean_s", ".3f", ""),
    ("SumRotationsDegThree", "get", "m1_latency_breakdown.json", "/by_k/k=3/timing_per_patient/sum_rotations_s/mean_s", ".3f", ""),
    ("SumRotationsDegFive", "get", "m1_latency_breakdown.json", "/by_k/k=5/timing_per_patient/sum_rotations_s/mean_s", ".3f", ""),
    ("SumRotationsDegSeven", "get", "m1_latency_breakdown.json", "/by_k/k=7/timing_per_patient/sum_rotations_s/mean_s", ".3f", ""),
    ("BreakdownWallDegThree", "get", "m1_latency_breakdown.json", "/by_k/k=3/timing_per_patient/wall_clock_total_s/mean_s", ".3f", ""),
    ("BreakdownWallDegFive", "get", "m1_latency_breakdown.json", "/by_k/k=5/timing_per_patient/wall_clock_total_s/mean_s", ".3f", ""),
    ("BreakdownWallDegSeven", "get", "m1_latency_breakdown.json", "/by_k/k=7/timing_per_patient/wall_clock_total_s/mean_s", ".3f", ""),
    ("GaloisKeyBytesDegFive", "get", "m1_latency_breakdown.json", "/by_k/k=5/key_sizes_bytes/galois_keys_bytes", ",", ""),
    ("EvalContextBytesDegFive", "get", "m1_latency_breakdown.json", "/by_k/k=5/key_sizes_bytes/evaluation_context_with_galois_keys_bytes", ",", ""),
    ("InputCiphertextBytesDegFive", "get", "m1_latency_breakdown.json", "/by_k/k=5/ciphertext_bytes/input_total_per_patient", ",", ""),

    # ---- Scaling (guide Sec 17) ----------------------------------------------------------
    ("ScalingMaxCell", "max", "m1_scaling_multiseed.json", "/per_cell/*/abs_delta_f1", ".4f",
     "SIMULATED; largest |dF1| cell across the whole 5-seed x 8-configuration scaling grid"),
    ("MemEvalPhaseMinKiB", "min", "m1_scaling_memory.json", "/per_configuration/*/memory_kb/evaluation_phase_increment_over_pre_evaluation_rss", ",",
     "REAL; one fresh process per configuration, seed 42, k=5"),
    ("MemEvalPhaseMaxKiB", "max", "m1_scaling_memory.json", "/per_configuration/*/memory_kb/evaluation_phase_increment_over_pre_evaluation_rss", ",", ""),
    ("MemKeysMinKiB", "min", "m1_scaling_memory.json", "/per_configuration/*/memory_kb/context_and_keys_increment", ",", ""),
    ("MemKeysMaxKiB", "max", "m1_scaling_memory.json", "/per_configuration/*/memory_kb/context_and_keys_increment", ",", ""),

    # ---- Security (guide Sec 18) ---------------------------------------------------------
    ("SecurityBitsMatzov", "get", "m1_lwe_security_summary.json", "/models/MATZOV (estimator default)/locked_chain_bits", ".1f", ""),
    ("SecurityBitsCoreSvp", "get", "m1_lwe_security_summary.json", "/models/ADPS16 classical core-SVP (HE Standard basis)/locked_chain_bits", ".1f", ""),
    ("EstimatorCommit", "get", "m1_lwe_security_summary.json", "/estimator_commit", None, ""),

    # ---- TFHE calibration reference (guide Sec 20) ----------------------------------------
    ("TfheLatencyMean", "get", "m1_tfhe_calibration.json", "/real_fhe_execution/online_latency_seconds_mean", ".1f", "REAL FHE execution."),
    ("TfheLatencySD", "get", "m1_tfhe_calibration.json", "/real_fhe_execution/online_latency_seconds_std", ".1f", ""),
    ("TfheNExec", "get", "m1_tfhe_calibration.json", "/real_fhe_execution/n_real_fhe_executions", None, ""),
    ("TfheAgreement", "get", "m1_tfhe_calibration.json", "/real_fhe_execution/agreement_tfhe_vs_quantized_clear", ".4f", ""),
]


E2 = os.environ.get("M1_E2_ARTIFACT", "m1_real_full_split.json")
if os.path.exists(os.path.join(_HERE, "..", "results", E2)):   # appended only once the full-split run has finished
    MACROS += [    # ---- Revision E2: the full test split on real ciphertexts (original evaluator, k=3, seed 42) --------------
        ("RealFullNStays", "get", E2, "/configuration/n_test_stays", ",", "REAL; one seed, one degree."),
        ("RealFullNEvals", "get", E2, "/n_evaluations", ",", "REAL ciphertext evaluations, all five labels."),
        ("RealFullNFailed", "get", E2, "/n_failed", None, "REAL."),
        ("RealFullFOnePlain", "get", E2, "/macro_f1/plaintext", ".4f", "Macro-F1 over the test split, plaintext."),
        ("RealFullFOneSim", "get", E2, "/macro_f1/simulated", ".4f", "SIMULATED."),
        ("RealFullFOneReal", "get", E2, "/macro_f1/real", ".4f", "REAL ciphertexts."),
        ("RealFullDeltaFOne", "get", E2, "/delta_f1_real_vs_plaintext/point_estimate", ".5f", "REAL deviation from plaintext."),
        ("RealFullDeltaLo", "get", E2, "/delta_f1_real_vs_plaintext/ci_95_lower", ".5f", ""),
        ("RealFullDeltaHi", "get", E2, "/delta_f1_real_vs_plaintext/ci_95_upper", ".5f", ""),
        ("RealFullSimDeltaFOne", "get", E2, "/delta_f1_simulated_vs_plaintext/point_estimate", ".5f", "SIMULATED."),
        ("RealFullSignAgree", "get", E2, "/agreement_all_evaluations/n_sign_agree_real_vs_simulated", ",", "REAL vs SIMULATED."),
        ("RealFullDecisionDisagree", "get", E2, "/agreement_all_evaluations/n_decision_disagree_real_vs_simulated_at_threshold", ",", "REAL vs SIMULATED at the frozen thresholds."),
        ("RealFullMaxErr", "get", E2, "/agreement_all_evaluations/max_abs_margin_error_real_vs_simulated", "sci2", "REAL vs SIMULATED."),
        ("RealFullMeanErr", "get", E2, "/agreement_all_evaluations/mean_abs_margin_error_real_vs_simulated", "sci2", "REAL vs SIMULATED."),
    ]

# ---- Continuous-feature task (added analysis; GATES.md Analysis-plan gate). Appended only once the final run exists. -------------
CTS, CTR, CTA = "m1c_final_summary.json", "m1c_final_real.json", "m1c_final_acuity_decision.json"
if os.path.exists(os.path.join(_HERE, "..", "results", CTS)):
    MACROS += [
        ("CTAcuityGain", "get", CTA, "/auroc_gain", ".3f", "Validation AUROC gain from adding triage acuity (5-bin model, seed 42)."),
        ("CTAurocWithAcuity", "get", CTA, "/results/with_acuity/auroc_val", ".3f", "Validation, seed 42."),
        ("CTAurocWithoutAcuity", "get", CTA, "/results/without_acuity/auroc_val", ".3f", "Validation, seed 42."),
        ("CTCohortStays", "get", CTS, "/cohort_flow/n_with_a_triage_vital", ",", "Adult ED stays with a recorded triage vital sign."),
        ("CTAdultStays", "get", CTS, "/cohort_flow/n_adult", ",", "Adult ED stays before the triage-vital requirement."),
        ("CTAurocCostLo", "get", CTS, "/quantization_cost/auroc_boot95_per_seed/0/0", ".4f", "Patient-cluster bootstrap, seed 42 (first seed)."),
        ("CTAurocCostHi", "get", CTS, "/quantization_cost/auroc_boot95_per_seed/0/1", ".4f", "Patient-cluster bootstrap, seed 42 (first seed)."),
        ("CTTestStays", "get", CTS, "/n_test_stays", ",", "Test split, continuous-feature task."),
        ("CTTestSubjects", "get", CTS, "/n_test_subjects", ",", ""),
        ("CTNSeeds", "get", CTS, "/n_seeds", None, ""),
        ("CTNInputs", "len", CTS, "/inputs/*", None, "Inputs of the continuous-feature task."),
        ("CTBinsPrimary", "get", CTS, "/exact_B5/bins", None, ""),
        ("CTDegreePrimary", "get", CTS, "/exact_B5/degree", None, ""),
        ("CTBinsSens", "get", CTS, "/exact_B8/bins", None, ""),
        ("CTDegreeSens", "get", CTS, "/exact_B8/degree", None, ""),
        ("CTFullAuroc", "get", CTS, "/plaintext_full_precision/auroc/mean", ".4f", "Full-precision plaintext model, test."),
        ("CTFullAurocSD", "get", CTS, "/plaintext_full_precision/auroc/sd", ".4f", ""),
        ("CTQuantAuroc", "get", CTS, "/plaintext_quantized_B5/auroc/mean", ".4f", "5-bin plaintext model, test."),
        ("CTQuantAurocSD", "get", CTS, "/plaintext_quantized_B5/auroc/sd", ".4f", ""),
        ("CTEightAuroc", "get", CTS, "/plaintext_quantized_B8/auroc/mean", ".4f", "8-bin plaintext model, test."),
        ("CTAurocCost", "get", CTS, "/quantization_cost/auroc/mean", ".4f", "Full-precision AUROC minus 5-bin AUROC, test, mean over seeds."),
        ("CTFullFOne", "get", CTS, "/plaintext_full_precision/f1op_f1/mean", ".3f", "F1 at the F1-maximizing threshold."),
        ("CTQuantFOne", "get", CTS, "/plaintext_quantized_B5/f1op_f1/mean", ".3f", ""),
        ("CTFullSensEighty", "get", CTS, "/plaintext_full_precision/sens80_specificity/mean", ".3f", "Specificity at the 80% sensitivity threshold."),
        ("CTQuantSensEighty", "get", CTS, "/plaintext_quantized_B5/sens80_specificity/mean", ".3f", ""),
        ("CTQuantPpvEighty", "get", CTS, "/plaintext_quantized_B5/sens80_ppv/mean", ".3f", ""),
        ("CTExactDifferingMax", "get", CTS, "/exact_B5/f1/n_differing_max_over_seeds", ",", "Most differing decisions in any seed, F1 operating point."),
        ("CTExactDifferingMaxSens", "get", CTS, "/exact_B5/sens80/n_differing_max_over_seeds", ",", "Most differing decisions in any seed, 80% sensitivity point."),
        ("CTExactMaxMargin", "max", CTS, "/exact_B5/f1/max_abs_margin_difference/per_seed/*", "sci2", "Largest simulated margin difference over seeds."),
        ("CTEightDifferingMax", "get", CTS, "/exact_B8/f1/n_differing_max_over_seeds", ",", "8-bin simulation."),
        ("CTFloorDiffDegOne", "get", CTS, "/floor/k=1/f1/differing_per_1000/mean", ".1f", "Approximate comparison, full precision."),
        ("CTFloorDiffDegThree", "get", CTS, "/floor/k=3/f1/differing_per_1000/mean", ".1f", ""),
        ("CTFloorDiffDegFive", "get", CTS, "/floor/k=5/f1/differing_per_1000/mean", ".1f", ""),
        ("CTFloorDiffDegSeven", "get", CTS, "/floor/k=7/f1/differing_per_1000/mean", ".1f", ""),
        ("CTFloorDiffDegNine", "get", CTS, "/floor/k=9/f1/differing_per_1000/mean", ".1f", ""),
        ("CTFloorDiffDegNineSD", "get", CTS, "/floor/k=9/f1/differing_per_1000/sd", ".1f", ""),
        ("CTFloorLostDegThree", "get", CTS, "/floor/k=3/f1/lost_share_of_plain_positives/mean", ".1%", ""),
        ("CTFloorLostDegSeven", "get", CTS, "/floor/k=7/f1/lost_share_of_plain_positives/mean", ".1%", ""),
        ("CTFloorLostDegNine", "get", CTS, "/floor/k=9/f1/lost_share_of_plain_positives/mean", ".1%", ""),
        ("CTFloorLostDegNineSD", "get", CTS, "/floor/k=9/f1/lost_share_of_plain_positives/sd", ".1%", ""),
        ("CTFloorAurocDeltaDegNine", "get", CTS, "/floor/k=9/auroc_delta/mean", ".4f", "Plaintext minus circuit AUROC."),
        ("CTFloorAurocDeltaDegOne", "get", CTS, "/floor/k=1/auroc_delta/mean", ".4f", "Plaintext minus circuit AUROC at k=1 (supervisor review 2026-10-14: the range over degrees)."),
        ("CTFloorAurocDeltaMax", "max", CTS, "/floor/*/auroc_delta/mean", ".4f", "Largest plaintext-minus-circuit AUROC over the degrees k=1,3,5,7,9 (seed mean)."),
        ("CTPilotFloorDegSixtyThree", "get", "m1c_pilot4.json", "/results/k=63/disagree_per_1000", ".1f", "Pilot, validation, seed 42."),
        ("CTPilotFloorLostDegSixtyThree", "get", "m1c_pilot4.json", "/results/k=63/lost_share_of_plain_positives", ".1%", "Pilot, validation, seed 42."),
        ("CTPilotFloorDegFifteen", "get", "m1c_pilot3.json", "/results/soft_step/k=15/disagree_per_1000", ".1f", "Pilot, validation, seed 42."),
        ("CTPilotAdjacencyShare", "get", "m1c_pilot4.json", "/adjacency/share_within_1_step", ".1%", "Pilot, validation, seed 42: share of comparisons within one measurement step of their threshold."),
        ("CTPilotAdjacencyRatio", "get", "m1c_pilot4.json", "/adjacency/ratio_adjacency_to_k63_disagreement", ".1f", "Pilot, validation, seed 42: that share divided by the degree-63 disagreement rate."),
        ("CTMaxCoefFive", "get", "m1c_pilot7.json", "/levels/B=5/max_abs_power_coefficient", ",.0f", "Largest power-basis coefficient of the 5-bin exact polynomial."),
        ("CTMaxCoefEight", "get", "m1c_pilot7.json", "/levels/B=8/max_abs_power_coefficient", ",.0f", "Largest power-basis coefficient of the 8-bin exact polynomial."),
        ("CTPolyvalErrFifteen", "get", "m1c_pilot7.json", "/tenseal_polyval_depth_probe/k=15/max_abs_error_vs_numpy", "sci2", "Synthetic input, TenSEAL polyval, power basis, degree 15."),
        ("CTPolyvalLevelsFifteen", "get", "m1c_pilot7.json", "/tenseal_polyval_depth_probe/k=15/levels_used_by_polyval", None, "Synthetic input."),
    ]
CTT = "m1c_final_ties.json"
if os.path.exists(os.path.join(_HERE, "..", "results", CTT)):
    MACROS += [
        ("CTTiesDiffering", "sum", CTT, "/per_seed/*/sens80/n_differing", ",", "Added analysis: differing decisions at the 80% sensitivity threshold, summed over seeds."),
        ("CTTiesAreTies", "sum", CTT, "/per_seed/*/sens80/n_differing_that_are_ties", ",", "Added analysis: of those, stays whose plaintext margin equals the threshold."),
        ("CTTiesTieFreeDiffering", "sum", CTT, "/per_seed/*/sens80_tie_free/n_differing", ",", "Added analysis: differing decisions at a threshold between attained margins, summed over seeds."),
        ("CTTiesSensTieFree", "mean", CTT, "/per_seed/*/sens80_tie_free/sensitivity_test_at_tie_free_threshold", ".3f", "Added analysis: test sensitivity at that threshold, mean over seeds."),
    ]
CTD = "m1c_final_ties_detail.json"
if os.path.exists(os.path.join(_HERE, "..", "results", CTD)):
    MACROS += [
        ("CTDetailMaxDiff", "get", CTD, "/max_abs_margin_difference_all_seeds", "sci2", "SIMULATED: largest absolute difference between simulated and plaintext margin on any test stay in any seed (added analysis)."),
        ("CTDetailPosToNeg", "sum", CTD, "/per_seed/*/n_differing_plain_positive_circuit_negative", ",", "SIMULATED: differing decisions at the 80% sensitivity threshold that the plaintext model called positive and the circuit negative, summed over seeds (added analysis)."),
        ("CTDetailNegToPos", "sum", CTD, "/per_seed/*/n_differing_plain_negative_circuit_positive", ",", "SIMULATED: the reverse direction, summed over seeds (added analysis)."),
        ("CTDetailDiffSeedA", "get", CTD, "/per_seed/123/min_signed_difference_among_differing", "sci2", "SIMULATED: signed difference simulated minus plaintext margin of the differing stays, first seed that had any (added analysis)."),
        ("CTDetailDiffSeedB", "get", CTD, "/per_seed/7/min_signed_difference_among_differing", "sci2", "SIMULATED: same, second seed that had any (added analysis)."),
    ]
CTRPATH = os.path.join(_HERE, "..", "results", CTR)
if os.path.exists(CTRPATH):
    MACROS += [
        ("CTRealStays", "get", CTR, "/n_stays", ",", "REAL ciphertexts, 5-bin exact circuit."),
        ("CTRealAgree", "get", CTR, "/decisions_agreeing_with_booster", ",", "REAL."),
        ("CTRealMaxErr", "get", CTR, "/max_abs_real_minus_sim", "sci2", "REAL vs SIMULATED."),
        ("CTRealGate", "get", CTR, "/gate_tolerance", "sci2", ""),
        ("CTRealSeconds", "get", CTR, "/mean_seconds_per_stay", ".2f", "REAL; one label."),
        ("CTRealSecondsSD", "get", CTR, "/sd_seconds_per_stay", ".2f", ""),
    ]


TMS = "m1c_thermo_final_summary.json"
if os.path.exists(os.path.join(_HERE, "..", "results", TMS)):
    _W = {5: "Five", 8: "Eight", 16: "Sixteen"}
    _CIRC = {"Bcast": "broadcast_thermometer_4level", "Pack": "packed_thermometer_4level", "Base": "client_indicator_baseline_4level"}
    MACROS += [
        ("TMNSeeds", "get", TMS, "/n_seeds", None, "Model-private thermometer protocol, five seeds."),
        ("TMTestStays", "get", TMS, "/n_test_stays", ",", "Test split, Proxy cohort (MIMIC-IV-ED-derived)."),
        ("TMRealStaysPerSeed", "get", TMS, "/n_real_stays_per_seed_per_B", None, "REAL ciphertexts: stays per seed and bin count."),
        ("TMFullAuroc", "get", TMS, "/full_precision/auroc_test/mean", ".4f", "Full-precision plaintext model, test, mean over seeds."),
        ("TMFullAurocSD", "get", TMS, "/full_precision/auroc_test/sd", ".4f", ""),
        ("TMFullFOne", "get", TMS, "/full_precision/f1_f1_op/mean", ".3f", "F1 at the midpoint F1 operating point."),
        ("TMSimMaxMargin", "max", TMS, "/by_B/*/thermometer_simulated/max_abs_margin_difference_over_seeds", "sci2", "SIMULATED: largest margin difference over seeds and bin counts."),
        ("TMNearMaxErr", "max", TMS, "/by_B/*/near_threshold/max_abs_margin_error_over_repetitions", "sci2", "REAL ciphertexts: largest margin error over the near-threshold repetitions."),
        ("TMNearRepsLarge", "get", TMS, "/by_B/B=5/near_threshold/f1/repetitions_per_seed/0", None, "REAL ciphertexts: repetitions per near-threshold stay, seed 42."),
        ("TMNearRepsSmall", "get", TMS, "/by_B/B=5/near_threshold/f1/repetitions_per_seed/1", None, "REAL ciphertexts: repetitions per near-threshold stay, the other seeds."),
        ("TMChainLevelsFour", "get", TMS, "/chain_levels/four_level", None, "Data levels of the four-level chain (REAL runs)."),
        ("TMChainLevelsLocked", "get", TMS, "/chain_levels/locked_seven_level", None, "Data levels of the locked chain (REAL runs)."),
        ("TMKeysFourGaloisMB", "get", TMS, "/key_bytes_one_off/n16384_four_level/galois_keys_mb", ".1f", "One-off Galois keys, four-level chain, TenSEAL default set (REAL runs)."),
        ("TMKeysFourTotalMB", "get", TMS, "/key_bytes_one_off/n16384_four_level/total_mb", ".1f", "One-off public, relinearization and Galois keys, four-level chain (REAL runs)."),
        ("TMKeysLockedGaloisMB", "get", TMS, "/key_bytes_one_off/n16384_locked_seven_level/galois_keys_mb", ".1f", "One-off Galois keys, locked chain (REAL runs)."),
        ("TMKeysLockedTotalMB", "get", TMS, "/key_bytes_one_off/n16384_locked_seven_level/total_mb", ".1f", "One-off keys, locked chain (REAL runs)."),
        ("TMLockedStays", "get", TMS, "/locked_chain_arm/by_B/B=5/packed_thermometer_locked_chain/n_stays", None, "REAL ciphertexts, locked chain, seed 42 (added arm)."),
        ("TMInterpDiffSum", "sum", TMS, "/by_B/B=5/interpolation_simulated/n_differing_f1_op_per_seed/*", ",", "SIMULATED: exact interpolation with midpoint thresholds, differing decisions summed over seeds, F1 point."),
        ("TMInterpDiffSumSens", "sum", TMS, "/by_B/B=5/interpolation_simulated/n_differing_sens80_op_per_seed/*", ",", "SIMULATED: same, 80% sensitivity point."),
        ("TMInterpSec", "get", TMS, "/by_B/B=5/real/exact_interpolation_7level/seconds_per_stay/mean", ".2f", "REAL ciphertexts, exact interpolation, locked chain."),
        ("TMInterpSecSD", "get", TMS, "/by_B/B=5/real/exact_interpolation_7level/seconds_per_stay/sd", ".2f", "REAL ciphertexts."),
        ("TMInterpUploadMB", "get", TMS, "/by_B/B=5/real/exact_interpolation_7level/client_upload_mb", ".2f", "REAL ciphertexts, per-stay upload."),
        ("TMInterpMaxErr", "get", TMS, "/by_B/B=5/real/exact_interpolation_7level/max_abs_margin_error", "sci2", "REAL ciphertexts."),
    ]
    for B, w in _W.items():
        k = f"B={B}"
        MACROS += [
            ("TMAuroc" + w, "get", TMS, f"/by_B/{k}/quantized_plain/auroc_test/mean", ".4f", "Quantized plaintext model, test."),
            ("TMAurocSD" + w, "get", TMS, f"/by_B/{k}/quantized_plain/auroc_test/sd", ".4f", "Quantized plaintext model, test."),
            ("TMCost" + w, "get", TMS, f"/by_B/{k}/quantized_plain/auroc_cost_vs_full_precision/mean", ".4f", "Full-precision minus quantized AUROC."),
            ("TMCostSD" + w, "get", TMS, f"/by_B/{k}/quantized_plain/auroc_cost_vs_full_precision/sd", ".4f", "Full-precision minus quantized AUROC."),
            ("TMFOne" + w, "get", TMS, f"/by_B/{k}/quantized_plain/f1_f1_op/mean", ".3f", "F1 at the midpoint F1 operating point."),
            ("TMSensSpec" + w, "get", TMS, f"/by_B/{k}/quantized_plain/specificity_sens80_op/mean", ".3f", "Specificity at the 80% sensitivity point."),
            ("TMSensPpv" + w, "get", TMS, f"/by_B/{k}/quantized_plain/ppv_sens80_op/mean", ".3f", "PPV at the 80% sensitivity point."),
            ("TMSimDiff" + w, "sum", TMS, f"/by_B/{k}/thermometer_simulated/n_differing_f1_op_per_seed/*", ",", "SIMULATED: differing decisions summed over seeds, F1 point."),
            ("TMSimDiffSens" + w, "sum", TMS, f"/by_B/{k}/thermometer_simulated/n_differing_sens80_op_per_seed/*", ",", "SIMULATED: same, 80% sensitivity point."),
            ("TMNearMin" + w, "min", TMS, f"/by_B/{k}/thermometer_simulated/stays_near_either_threshold_per_seed/*", ",", "SIMULATED margins: stays within the gate of a threshold, smallest over seeds."),
            ("TMNearMax" + w, "max", TMS, f"/by_B/{k}/thermometer_simulated/stays_near_either_threshold_per_seed/*", ",", "SIMULATED margins: largest over seeds."),
            ("TMNearFoneStays" + w, "get", TMS, f"/by_B/{k}/near_threshold/f1/total_stays", ",", "REAL ciphertexts: near-threshold stays at the F1 threshold, summed over seeds."),
            ("TMNearFoneFlipped" + w, "get", TMS, f"/by_B/{k}/near_threshold/f1/total_flipped", ",", "REAL ciphertexts: of those, flipped in at least one repetition."),
            ("TMNearSensStays" + w, "get", TMS, f"/by_B/{k}/near_threshold/sens80/total_stays", ",", "REAL ciphertexts: near-threshold stays at the 80% sensitivity threshold."),
            ("TMNearSensFlipped" + w, "get", TMS, f"/by_B/{k}/near_threshold/sens80/total_flipped", ",", "REAL ciphertexts: of those, flipped in at least one repetition."),
            ("TMNearSimDiff" + w, "sum", TMS, f"/by_B/{k}/near_threshold/f1/n_simulated_differing_per_seed/*", ",", "SIMULATED decisions for the near-threshold stays, F1 threshold."),
            ("TMNearSimDiffSens" + w, "sum", TMS, f"/by_B/{k}/near_threshold/sens80/n_simulated_differing_per_seed/*", ",", "SIMULATED decisions for the near-threshold stays, 80% sensitivity."),
        ]
        for tag, c in _CIRC.items():
            r = f"/by_B/{k}/real/{c}"
            MACROS += [
                (f"TM{tag}Sec" + w, "get", TMS, f"{r}/seconds_per_stay/mean", ".2f", "REAL ciphertexts, seconds per stay."),
                (f"TM{tag}Up" + w, "get", TMS, f"{r}/client_upload_mb", ".2f", "REAL ciphertexts, per-stay upload in MB."),
                (f"TM{tag}Err" + w, "get", TMS, f"{r}/max_abs_margin_error", "sci2", "REAL ciphertexts, largest margin error."),
                (f"TM{tag}AgreeF" + w, "get", TMS, f"{r}/decisions_agreeing_f1_op", ",", "REAL ciphertexts, decisions agreeing at the F1 point."),
                (f"TM{tag}AgreeS" + w, "get", TMS, f"{r}/decisions_agreeing_sens80_op", ",", "REAL ciphertexts, decisions agreeing at the 80% sensitivity point."),
            ]
        MACROS += [("TMRealStays" + w, "get", TMS, f"/by_B/{k}/real/packed_thermometer_4level/n_stays_total", ",", "REAL ciphertexts, stays per circuit over seeds.")]
        for tag, c in (("LockBcast", "broadcast_thermometer_locked_chain"), ("LockPack", "packed_thermometer_locked_chain")):
            r = f"/locked_chain_arm/by_B/{k}/{c}"
            MACROS += [
                (f"TM{tag}Sec" + w, "get", TMS, f"{r}/mean_seconds_per_stay", ".2f", "REAL ciphertexts, locked chain, seed 42 (added arm)."),
                (f"TM{tag}Up" + w, "get", TMS, f"{r}/client_upload_mb", ".2f", "REAL ciphertexts, locked chain, per-stay upload in MB."),
            ]


SCS = "m1c_thermo_samechain_summary.json"
if os.path.exists(os.path.join(_HERE, "..", "results", SCS)):
    # Same-chain comparison (added study, five seeds, locked chain, REAL ciphertexts). Sample: the thermometer study's frozen stratified sample (50 + 50 at the F1 threshold).
    _SC_CIRC = {"Bcast": "broadcast", "Pack": "packed", "Interp": "interpolation"}
    _SC_W = {5: "Five", 8: "Eight", 16: "Sixteen"}
    for B, w in _SC_W.items():
        k = f"B={B}"
        for tag, c in _SC_CIRC.items():
            if c == "interpolation" and B != 5:
                continue
            r = f"/by_B/{k}/circuits/{c}"
            MACROS += [
                (f"SC{tag}Sec" + w, "get", SCS, f"{r}/mean_seconds/mean", ".2f", "REAL ciphertexts, same chain, mean over 5 seeds of per-seed mean seconds per stay."),
                (f"SC{tag}SecSD" + w, "get", SCS, f"{r}/mean_seconds/sd", ".2f", "REAL ciphertexts, same chain, SD over 5 seeds of the per-seed mean."),
                (f"SC{tag}Up" + w, "get", SCS, f"{r}/client_upload_mb", ".2f", "REAL ciphertexts, per-stay upload in MB."),
                (f"SC{tag}Err" + w, "get", SCS, f"{r}/max_abs_margin_error_over_seeds", "sci2", "REAL ciphertexts, largest margin error over 5 seeds."),
                (f"SC{tag}AgreeF" + w, "get", SCS, f"{r}/decisions_agreeing_f1_total", ",", "REAL ciphertexts, decisions agreeing at the F1 point (sample drawn around that threshold)."),
                (f"SC{tag}AgreeS" + w, "get", SCS, f"{r}/decisions_agreeing_sens80_total", ",", "REAL ciphertexts, decisions agreeing at the 80% sensitivity point (sample not drawn around it)."),
                (f"SC{tag}Stays" + w, "get", SCS, f"{r}/n_stays_total", ",", "REAL ciphertexts, stays per circuit over 5 seeds."),
            ]
        MACROS += [
            ("SCGapSampleF" + w, "min", SCS, f"/by_B/{k}/margin_gaps/*/f1/min_gap_sample_100", "sci2", "Margin gap, 100-stay samples, F1 threshold, smallest over seeds."),
            ("SCGapSampleS" + w, "min", SCS, f"/by_B/{k}/margin_gaps/*/sens80/min_gap_sample_100", "sci2", "Margin gap, 100-stay samples, 80% sensitivity threshold, smallest over seeds."),
            ("SCGapFullF" + w, "min", SCS, f"/by_B/{k}/margin_gaps/*/f1/min_gap_full_test_split", "sci2", "Margin gap, full test split, F1 threshold, smallest over seeds."),
            ("SCGapFullS" + w, "min", SCS, f"/by_B/{k}/margin_gaps/*/sens80/min_gap_full_test_split", "sci2", "Margin gap, full test split, 80% sensitivity threshold, smallest over seeds."),
            ("SCBandMinF" + w, "min", SCS, f"/by_B/{k}/margin_gaps/*/f1/n_full_split_within_gate", ",", "Full test split: stays within the gate of the F1 threshold, smallest over seeds."),
            ("SCBandMaxF" + w, "max", SCS, f"/by_B/{k}/margin_gaps/*/f1/n_full_split_within_gate", ",", "Full test split: stays within the gate of the F1 threshold, largest over seeds."),
            ("SCBandMinS" + w, "min", SCS, f"/by_B/{k}/margin_gaps/*/sens80/n_full_split_within_gate", ",", "Full test split: stays within the gate of the 80% threshold, smallest over seeds."),
            ("SCBandMaxS" + w, "max", SCS, f"/by_B/{k}/margin_gaps/*/sens80/n_full_split_within_gate", ",", "Full test split: stays within the gate of the 80% threshold, largest over seeds."),
        ]
    MACROS += [("SCSeeds", "len", SCS, "/seeds/*", None, "Seeds of the same-chain run.")]


FCS = "m1c_formal_counts.json"
if os.path.exists(os.path.join(_HERE, "..", "results", FCS)):
    # Counts derived by running the real evaluation functions on a depth-tracking mock (added analysis, src/m1c_formal_counts.py); saved model files only.
    _FC_W = {5: "Five", 8: "Eight", 16: "Sixteen"}
    for B, w in _FC_W.items():
        k = f"B={B}"
        d = f"/by_B/{k}/design"
        MACROS += [
            ("FCBitCols" + w, "get", FCS, f"{d}/bit_columns", None, "Derived: inputs x (B - 1), the number of thermometer bits."),
            ("FCCtBcast" + w, "get", FCS, f"{d}/client_ciphertexts/broadcast", None, "Derived by running encrypt_bits on a mock: client ciphertexts, broadcast."),
            ("FCCtInterp" + w, "get", FCS, f"{d}/client_ciphertexts/interpolation", None, "Derived: client ciphertexts, interpolation (one per tree level)."),
            ("FCLevBcast" + w, "get", FCS, f"{d}/levels/broadcast", None, "Derived: multiplicative depth, broadcast thermometer."),
            ("FCLevPack" + w, "get", FCS, f"{d}/levels/packed", None, "Derived: multiplicative depth, packed thermometer."),
            ("FCLevInterp" + w, "get", FCS, f"{d}/levels/interpolation", None, "Derived: depth of the interpolation circuit with the folding evaluator."),
            ("FCDegree" + w, "get", FCS, f"{d}/interpolation_degree", None, "Derived: degree 2B - 1 of the exact comparison polynomial."),
            ("FCRatioMeasured" + w, "get", FCS, f"/by_B/{k}/measured_upload_ratio_broadcast_to_packed", ".2f", "REAL ciphertexts: measured broadcast-to-packed upload ratio, from the same-chain run."),
            ("FCPackRotMin" + w, "min", FCS, f"/by_B/{k}/seeds/*/packed/rotation_calls", ",", "Derived: rotation calls of the packed circuit, smallest over seeds."),
            ("FCPackRotMax" + w, "max", FCS, f"/by_B/{k}/seeds/*/packed/rotation_calls", ",", "Derived: rotation calls of the packed circuit, largest over seeds."),
            ("FCPackRotMM" + w, "min", FCS, f"/by_B/{k}/seeds/*/packed/rotation_calls_in_matrix_product", ",", "Derived: rotation calls in the matrix products of the packed circuit (equal over seeds)."),
            ("FCBcastRot" + w, "min", FCS, f"/by_B/{k}/seeds/*/broadcast/rotation_calls", ",", "Derived: rotation calls of the broadcast circuit (the final sum)."),
            ("FCLeafMin" + w, "min", FCS, f"/by_B/{k}/seeds/*/leaf_slots", ",", "Leaf slots of the saved models, smallest over seeds."),
            ("FCLeafMax" + w, "max", FCS, f"/by_B/{k}/seeds/*/leaf_slots", ",", "Leaf slots of the saved models, largest over seeds."),
            ("FCInterpCtCt" + w, "min", FCS, f"/by_B/{k}/seeds/*/interpolation/ct_ct_multiplications", ",", "Derived: ciphertext-ciphertext multiplications of the interpolation circuit."),
            ("FCResid" + w, "get", FCS, f"/by_B/{k}/interpolant_check/max_node_residual", "sci2", "Largest residual of the double-precision interpolant at its nodes."),
            ("FCCond" + w, "get", FCS, f"/by_B/{k}/interpolant_check/condition_number", "sci2", "Condition number of the interpolation system in double precision."),
        ]

SDC = "m1c_security_design_constants.json"
if os.path.exists(os.path.join(_HERE, "..", "results", SDC)):
    MACROS += [
        ("SecBitsFive", "get", SDC, "/single_bit_probe_set_size/B=5", None, "Design constant: single-bit probe set of the thermometer protocol, n_inputs x (B - 1). Not an attack cost."),
        ("SecBitsEight", "get", SDC, "/single_bit_probe_set_size/B=8", None, "Design constant, as above."),
        ("SecBitsSixteen", "get", SDC, "/single_bit_probe_set_size/B=16", None, "Design constant, as above."),
    ]

SEC = "m1c_signerr_check.json"
if os.path.exists(os.path.join(_HERE, "..", "results", SEC)):
    # Sign of the margin error on the one stay that disagreed at 8 bins (added analysis; gates "M1 sign of the margin error"; src/m1c_signerr_check.py). One stay, no identifier.
    MACROS += [
        ("SEReps", "get", SEC, "/reps", None, "Fresh encryptions per circuit of the one stay."),
        ("SEGap", "get", SEC, "/stay_signed_gap_margin_minus_threshold", "sci2", "REAL check: the stay's margin minus the 80% threshold (positive: above)."),
        ("SEBcastMean", "get", SEC, "/broadcast/mean", "sci2", "REAL: mean signed error (decrypted minus exact margin), broadcast, 20 encryptions."),
        ("SEPackMean", "get", SEC, "/packed/mean", "sci2", "REAL: mean signed error, packed."),
        ("SEBcastSD", "get", SEC, "/broadcast/sd", "sci2", "REAL: SD of the signed error, broadcast."),
        ("SEPackSD", "get", SEC, "/packed/sd", "sci2", "REAL: SD of the signed error, packed."),
        ("SEBcastNeg", "get", SEC, "/broadcast/n_negative", None, "REAL: repetitions with a negative signed error, broadcast."),
        ("SEPackNeg", "get", SEC, "/packed/n_negative", None, "REAL: repetitions with a negative signed error, packed."),
        ("SEBcastToward", "get", SEC, "/broadcast/n_toward_threshold", None, "REAL: repetitions whose error pushes toward the threshold, broadcast."),
        ("SEPackToward", "get", SEC, "/packed/n_toward_threshold", None, "REAL: repetitions whose error pushes toward the threshold, packed."),
        ("SEBcastDiff", "get", SEC, "/broadcast/n_decision_differs_from_plaintext", None, "REAL: repetitions whose decision differs from the plaintext model, broadcast."),
        ("SEPackDiff", "get", SEC, "/packed/n_decision_differs_from_plaintext", None, "REAL: repetitions whose decision differs from the plaintext model, packed."),
    ]

THR = "m1c_threshold_integers.json"
if os.path.exists(os.path.join(_HERE, "..", "results", THR)):
    # Split thresholds of the saved thermometer-study boosters (added check, src/m1c_threshold_integers.py); counts only.
    MACROS += [
        ("ThrBoosters", "get", THR, "/n_boosters", None, "Saved boosters of the thermometer study (five seeds, three bin counts)."),
        ("ThrSplits", "get", THR, "/n_splits", ",", "Split nodes over those boosters."),
        ("ThrNonInt", "get", THR, "/n_non_integer_thresholds", None, "Split thresholds that are not integers."),
        ("ThrMin", "get", THR, "/smallest_threshold", ".0f", "Smallest split threshold."),
        ("ThrTolDiffers", "get", THR, "/n_splits_where_ceil_with_tolerance_differs_from_ceil", None, "Splits where ceil(t - 1e-9) differs from ceil(t)."),
    ]


def main() -> int:
    out_path, ev = emit_macro_file(MACROS, OUT_PATH)
    sidecar_path = ev.write_provenance()
    print(f"wrote {out_path} ({len(MACROS)} macros)")
    print(f"  provenance -> {sidecar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
