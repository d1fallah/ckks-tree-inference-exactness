#!/usr/bin/env python3
"""
300 DPI vector+raster figures from the real results JSONs, per CLAUDE.md Sec 6 and the Q1 Paper
Finalization Guide Sec 23 (data figures F3-F8; F1/F2 are conceptual diagrams built separately --
see paper_assets/figures_src/ and the note at the end of main()). Every point goes through
src/paper_data.py's Evidence accessor, exactly like the tables -- see that module's docstring.

Each figure writes BOTH .pdf and .png (300 DPI) into paper_assets/figures/, plus a provenance
sidecar in paper_assets/provenance/ (figures have no .tex for the token-scan check to read, so
their QC-label wording lives in the sidecar's `note=` fields instead -- see
verify_paper_assets.check_qc_labels).

Output: paper_assets/figures/*.{pdf,png} + paper_assets/provenance/*.json
Usage: python generate_figures.py
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_data import Evidence, real_latency_by_degree  # noqa: E402
import figstyle as fs  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

fs.apply()

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_HERE, "..", "paper_assets", "figures")

SIMULATED = "SIMULATED"
PROXY = "Proxy cohort (MIMIC-IV-ED-derived)"
K9_NOTE = ("the k=9 utility result is simulation only; the original evaluator cannot execute k=9 (8 levels needed, 7 available) "
           "and the coefficient-folding evaluator can")
FOLDED_ARTIFACT = "m1_encrypted_multiseed_folded.json"
K1_NOTE = "k=1 is an additional diagnostic arm outside the locked degree family"

# matplotlib PDF metadata includes a CreationDate by default, which makes two runs of the same
# figure byte-different even when every plotted value is identical -- pass None so re-generation
# is (as close as matplotlib allows) reproducible, matching the intent of the evidence freeze.
SAVE_KW = dict(dpi=300, bbox_inches="tight", metadata={"CreationDate": None})


def savefig(fig: "plt.Figure", asset_id: str, ev: Evidence) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(OUT_DIR, f"{asset_id}.{ext}")
        fig.savefig(path, **SAVE_KW)
        print(f"wrote {path}")
    plt.close(fig)
    sidecar_path = ev.write_provenance()
    print(f"  provenance -> {sidecar_path}")


# ---------------------------------------------------------------------------------------------
# Drawing helpers shared by the data figures. Every plotted value still comes through the Evidence
# accessor below; only the rendering changed in the 2026-10-09 redraw (figstyle.py has the rules).
# ---------------------------------------------------------------------------------------------
def _errbars(ax, x, y, yerr, color, cap=3.0, lw=1.1, zorder=3, min_frac=0.035):
    """Error bars; a bar shorter than ~3.5% of the axis span would sit inside the marker and render as
    a stray box, so it is omitted (every caption says "bars shorter than the marker are not drawn")."""
    lo, hi = ax.get_ylim()
    span = (hi - lo) or 1.0
    xs, ys, ls, us = [], [], [], []
    scalar = not isinstance(yerr[0], (list, tuple))
    for i, (xi, yi) in enumerate(zip(x, y)):
        e_lo = yerr[i] if scalar else yerr[0][i]
        e_hi = yerr[i] if scalar else yerr[1][i]
        if max(e_lo, e_hi) >= min_frac * span:
            xs.append(xi); ys.append(yi); ls.append(e_lo); us.append(e_hi)
    if xs:
        ax.errorbar(xs, ys, yerr=[ls, us], fmt="none", ecolor=color, elinewidth=lw, capsize=cap,
                    capthick=lw, zorder=zorder)


def _break_marks(ax_top, ax_bot):
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=7, linestyle="none", color=fs.MUTED,
              mec=fs.MUTED, mew=0.8, clip_on=False)
    ax_top.plot([0], [0], transform=ax_top.transAxes, **kw)
    ax_bot.plot([0], [1], transform=ax_bot.transAxes, **kw)


# ---------------------------------------------------------------------------------------------
# F3 -- Delta F1 vs polynomial degree, with the tolerance line. k=1 shown as an additional arm.
# The y-axis is broken so the tolerance (0.03) and the measured deviations (<= 0.0055 with their
# intervals) are both visible without the data being flattened against the axis.
# ---------------------------------------------------------------------------------------------
def fig_f3_delta_f1_vs_degree() -> None:
    ev = Evidence("f3_delta_f1_vs_degree")
    tolerance = ev.get("m1_ckks_sweep_simulated.json", "/utility_tolerance", fmt=None,
                        note=f"{SIMULATED}. {PROXY}. {K1_NOTE}. {K9_NOTE}.")

    degrees = [1, 3, 5, 7, 9]
    means, ci_lo, ci_hi = [], [], []
    for degree in degrees:
        if degree == 1:
            m = ev.get("m1_ckks_sweep_k1_simulated.json", "/delta_f1_mean_over_seeds")
            lo = ev.get("m1_ckks_sweep_k1_simulated.json", "/paired_bootstrap/ci_95_lower")
            hi = ev.get("m1_ckks_sweep_k1_simulated.json", "/paired_bootstrap/ci_95_upper")
        else:
            key = f"k={degree}"
            m = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/delta_f1_mean_over_seeds")
            lo = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/paired_bootstrap/ci_95_lower")
            hi = ev.get("m1_ckks_sweep_simulated.json", f"/k_sweep/{key}/paired_bootstrap/ci_95_upper")
        means.append(m)
        ci_lo.append(lo)
        ci_hi.append(hi)

    fig = plt.figure(figsize=(fs.W_COL, 2.9))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.2, 1.0], hspace=0.09)
    ax_t = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_t)

    # upper strip: the tolerance only
    ax_t.set_ylim(tolerance * 0.93, tolerance * 1.07)
    ax_t.axhline(tolerance, color=fs.INK2, linewidth=1.0, zorder=2)
    ax_t.set_yticks([tolerance])
    ax_t.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax_t.spines["bottom"].set_visible(False)
    ax_t.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_t.text(9.55, tolerance * 1.015, f"tolerance {tolerance}", ha="right", va="bottom",
              fontsize=8, color=fs.INK2)

    # lower panel: the measured deviations with their 95% intervals
    ymax = max(ci_hi) * 1.22
    ax_b.set_ylim(0, ymax)
    ax_b.yaxis.set_major_locator(MaxNLocator(4, prune=None))
    ax_b.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    fs.value_grid(ax_b)
    e3 = [[m - lo for m, lo in zip(means[1:], ci_lo[1:])], [hi - m for m, hi in zip(means[1:], ci_hi[1:])]]
    ax_b.plot(degrees[1:], means[1:], color=fs.ORIG, linewidth=1.6, zorder=3)
    _errbars(ax_b, degrees[1:], means[1:], e3, fs.ORIG)
    ax_b.plot(degrees[1:], means[1:], "o", color=fs.ORIG, **fs.ringed())
    # k = 1: additional diagnostic arm, hollow marker in neutral ink
    e1 = [[means[0] - ci_lo[0]], [ci_hi[0] - means[0]]]
    _errbars(ax_b, [1], [means[0]], e1, fs.INK2)
    ax_b.plot([1], [means[0]], "D", color=fs.INK2, mfc=fs.SURFACE, mec=fs.INK2, mew=1.4, markersize=6.5, zorder=4)
    ax_b.text(1, ci_hi[0] + ymax * 0.035, "$k=1$ diagnostic arm", ha="left", va="bottom", fontsize=8, color=fs.INK2)
    for x, m, hi in ((3, means[1], ci_hi[1]), (9, means[4], ci_hi[4])):
        ax_b.text(x, hi + ymax * 0.035, f"{m:.4f}", ha="center", va="bottom", fontsize=8, color=fs.INK2)

    _break_marks(ax_t, ax_b)
    ax_b.set_xlim(0.4, 9.6)
    ax_b.set_xticks(degrees)
    ax_b.set_xlabel("comparison-polynomial degree $k$")
    fig.supylabel(r"macro-F1 deviation $\Delta$F1 (simulated)", x=0.01, fontsize=9)
    savefig(fig, "f3_delta_f1_vs_degree", ev)


# ---------------------------------------------------------------------------------------------
# F4 -- Real latency vs polynomial degree (k=9 has no real point for the original evaluator)
# ---------------------------------------------------------------------------------------------
def fig_f4_real_latency_vs_degree() -> None:
    ev = Evidence("f4_real_latency_vs_degree")
    ev.get("m1_encrypted_multiseed.json", "/backend", note=f"REAL ciphertext execution. {PROXY}. {K9_NOTE}.")
    lat_o = real_latency_by_degree(ev)
    lat_f = real_latency_by_degree(ev, FOLDED_ARTIFACT)

    fig, ax = plt.subplots(figsize=(fs.W_COL, 2.9))
    fs.value_grid(ax)
    ax.set_ylim(0, 2.5)
    for data, color, marker, label in ((lat_o, fs.ORIG, "o", "original evaluator"),
                                       (lat_f, fs.FOLD, "s", "coefficient-folding evaluator")):
        degrees = sorted(data)
        mean = [data[d][0] for d in degrees]
        sd = [data[d][1] for d in degrees]
        ax.plot(degrees, mean, color=color, linewidth=1.6, zorder=3)
        _errbars(ax, degrees, mean, sd, color)
        ax.plot(degrees, mean, marker, color=color, label=label, **fs.ringed())
        ax.text(degrees[-1], mean[-1] + 0.1, f"{mean[-1]:.2f} s", ha="center", va="bottom", fontsize=8, color=fs.INK2)
    ax.text(9.45, 0.12, "original evaluator at $k=9$:\n8 levels needed, 7 available", ha="right", va="bottom",
            fontsize=7.5, color=fs.MUTED, linespacing=1.15)
    ax.set_xlabel("comparison-polynomial degree $k$")
    ax.set_ylabel("latency per patient (s, real ciphertexts)")
    ax.set_xticks([3, 5, 7, 9])
    ax.set_xlim(2.4, 9.6)
    ax.set_ylim(0, 2.5)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="upper left", handlelength=1.6, borderaxespad=0.2)
    fig.tight_layout()
    savefig(fig, "f4_real_latency_vs_degree", ev)


# ---------------------------------------------------------------------------------------------
# F5/F6 -- Tree-count and tree-depth scaling: two stacked panels sharing x (no dual axis). The two
# figures use the same y-scales so they can be compared directly.
# ---------------------------------------------------------------------------------------------
DF1_YMAX = 0.0135
LAT_YMAX = 1.05


def _scaling_panels(cells, agg, xkey, fixed, xlabel, ev: Evidence, series_note: str):
    rows = sorted([r for r in agg if fixed(r)], key=lambda r: r[xkey])
    xs = [r[xkey] for r in rows]
    ev.series("m1_scaling_multiseed.json", "/aggregated_across_seeds/*/abs_delta_f1/mean", note=series_note)

    fig, (ax_u, ax_l) = plt.subplots(2, 1, sharex=True, figsize=(fs.W_COL, 3.7), gridspec_kw={"hspace": 0.14})
    # utility: one dot per seed behind mean +- SD
    seed_x = [c[xkey] for c in cells if fixed(c)]
    seed_y = [c["abs_delta_f1"] for c in cells if fixed(c)]
    fs.value_grid(ax_u)
    ax_u.set_ylim(0, DF1_YMAX)
    ax_l.set_ylim(0, LAT_YMAX)
    ax_u.plot(seed_x, seed_y, "o", color=fs.ORIG, alpha=0.32, markersize=3.6, markeredgewidth=0, zorder=2)
    means = [r["abs_delta_f1"]["mean"] for r in rows]
    sds = [r["abs_delta_f1"]["sd_across_seeds"] for r in rows]
    ax_u.plot(xs, means, color=fs.ORIG, linewidth=1.6, zorder=3)
    _errbars(ax_u, xs, means, sds, fs.ORIG)
    ax_u.plot(xs, means, "o", color=fs.ORIG, **fs.ringed())
    ax_u.set_ylim(0, DF1_YMAX)
    ax_u.set_yticks([0, 0.004, 0.008, 0.012])
    ax_u.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_u.set_ylabel(r"$|\Delta$F1$|$ (simulated)")
    handles = [Line2D([], [], marker="o", color=fs.ORIG, alpha=0.4, linestyle="none", markersize=4.5, markeredgewidth=0),
               Line2D([], [], marker="o", color=fs.ORIG, linestyle="-", markersize=6, markeredgecolor=fs.SURFACE)]
    ax_u.legend(handles, ["one seed", "mean $\\pm$ SD over seeds"], loc="upper right", handlelength=1.6,
                borderaxespad=0.2, fontsize=8)

    # latency: mean +- SD, real ciphertexts
    fs.value_grid(ax_l)
    lat_means = [r["real_latency_wall_clock_s"]["mean"] for r in rows]
    lat_sds = [r["real_latency_wall_clock_s"]["sd_across_seeds"] for r in rows]
    ax_l.plot(xs, lat_means, color=fs.ORIG, linewidth=1.6, zorder=3)
    _errbars(ax_l, xs, lat_means, lat_sds, fs.ORIG)
    ax_l.plot(xs, lat_means, "o", color=fs.ORIG, **fs.ringed())
    for x, m in ((xs[0], lat_means[0]), (xs[-1], lat_means[-1])):
        ax_l.text(x, m + 0.07, f"{m:.2f} s", ha="center", va="bottom", fontsize=8, color=fs.INK2)
    ax_l.set_ylim(0, LAT_YMAX)
    ax_l.yaxis.set_major_locator(MaxNLocator(5))
    ax_l.set_ylabel("latency per patient (s)")
    ax_l.set_xticks(xs)
    ax_l.set_xlabel(xlabel)
    pad = (xs[-1] - xs[0]) * 0.07
    ax_l.set_xlim(xs[0] - pad, xs[-1] + pad)
    fig_ax = (fig, ax_u, ax_l)
    return fig_ax


def fig_f5_tree_count_scaling() -> None:
    ev = Evidence("f5_tree_count_scaling")
    cells = ev.series("m1_scaling_multiseed.json", "/per_cell/*",
                       note=f"{SIMULATED} utility, REAL latency. {PROXY}. k=5, one label, 5 seeds.")
    agg = ev.series("m1_scaling_multiseed.json", "/aggregated_across_seeds/*")
    fig, _, _ = _scaling_panels(cells, agg, "n_estimators", lambda r: r["max_depth"] == 3,
                                "number of trees (maximum depth 3)", ev, "tree-count scaling series")
    savefig(fig, "f5_tree_count_scaling", ev)


def fig_f6_tree_depth_scaling() -> None:
    ev = Evidence("f6_tree_depth_scaling")
    cells = ev.series("m1_scaling_multiseed.json", "/per_cell/*",
                       note=f"{SIMULATED} utility, REAL latency. {PROXY}. k=5, one label, 5 seeds.")
    agg = ev.series("m1_scaling_multiseed.json", "/aggregated_across_seeds/*")
    fig, _, _ = _scaling_panels(cells, agg, "max_depth", lambda r: r["n_estimators"] == 50,
                                "maximum tree depth (50 trees)", ev, "tree-depth scaling series")
    savefig(fig, "f6_tree_depth_scaling", ev)


# ---------------------------------------------------------------------------------------------
# F7 -- Multiplicative-depth / feasibility for k = 1,3,5,7,9, both evaluators side by side.
# Colour = evaluator (same as every other figure); dark tint = polynomial levels, light tint =
# path-product levels. Levels beyond the chain are drawn in the reserved status colour and labelled.
# ---------------------------------------------------------------------------------------------
def fig_f7_depth_feasibility() -> None:
    ev = Evidence("f7_depth_feasibility")
    available = ev.get("m1_depth_probe.json", "/per_degree/k=3/levels_available_on_locked_chain",
                        note=f"{K1_NOTE}. {K9_NOTE}. {PROXY}.")
    groups = []  # (label, [(evaluator, poly, path, fits), ...])
    for degree in (1, 3, 5, 7, 9):
        if degree == 1:
            poly = ev.get("m1_depth_budget_k1.json", "/code_derived/polynomial_depth")
            path = ev.get("m1_depth_budget_k1.json", "/code_derived/path_product_depth")
            fit = ev.get("m1_depth_budget_k1.json", "/fits_locked_chain")
            groups.append(("$k=1$ (diagnostic)", [("orig", poly, path, fit)]))
            continue
        key = f"k={degree}"
        poly = ev.get("m1_depth_probe.json", f"/per_degree/{key}/code_derived/polynomial_depth")
        path = ev.get("m1_depth_probe.json", f"/per_degree/{key}/code_derived/path_product_depth")
        fit = ev.get("m1_depth_probe.json", f"/per_degree/{key}/fits_locked_chain")
        poly_f = ev.get("m1_folded_depth_accounting.json", f"/per_degree/{key}/polynomial_depth_folded")
        total_f = ev.get("m1_folded_depth_accounting.json", f"/per_degree/{key}/total_depth_folded")
        fit_f = ev.get("m1_folded_depth_accounting.json", f"/per_degree/{key}/fits_folded")
        groups.append((f"$k={degree}$", [("orig", poly, path, fit), ("fold", poly_f, total_f - poly_f, fit_f)]))

    fig, ax = plt.subplots(figsize=(fs.W_COL, 4.1))
    fig.subplots_adjust(top=0.92, left=0.2, right=0.97, bottom=0.12)
    bar_h, dy = 0.3, 0.17
    colors = {"orig": (fs.ORIG, fs.ORIG_LT), "fold": (fs.FOLD, fs.FOLD_LT)}
    ytick, ylab = [], []
    fs.value_grid(ax, axis="x")
    for gi, (label, bars) in enumerate(groups):
        yc = len(groups) - 1 - gi
        offsets = [0.0] if len(bars) == 1 else [dy, -dy]
        for (which, poly, path, fits), off in zip(bars, offsets):
            y = yc + off
            dark, light = colors[which]
            total = poly + path
            ax.barh(y, poly, height=bar_h, color=dark, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
            if fits:
                ax.barh(y, path, left=poly, height=bar_h, color=light, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
                ax.text(total + 0.12, y, f"{total}", va="center", ha="left", fontsize=8, color=fs.INK2)
            else:
                fit_part = max(available - poly, 0)
                ax.barh(y, fit_part, left=poly, height=bar_h, color=light, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
                ax.barh(y, total - available, left=available, height=bar_h, color=fs.CRIT_LT, edgecolor=fs.SURFACE,
                        linewidth=1.2, zorder=3)
                ax.plot([total + 0.28], [y], "X", color=fs.CRIT, markersize=6, markeredgecolor=fs.SURFACE,
                        markeredgewidth=0.8, zorder=5)
                ax.text(total + 0.5, y, f"{total}: one level over the limit", va="center", ha="left", fontsize=8,
                        color=fs.INK2)
        ytick.append(yc)
        ylab.append(label)
    ax.axvline(available, color=fs.INK, linewidth=1.1, zorder=4)
    ax.text(available - 0.12, len(groups) - 0.42, f"chain limit: {available} levels", ha="right", va="bottom",
            fontsize=8, color=fs.INK)
    # key for the two tints, drawn as a small example bar in the free upper-right area
    kx, ky = 8.0, len(groups) - 1.0
    ax.barh(ky, 1, left=kx, height=0.26, color=fs.INK2, edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    ax.barh(ky, 1.6, left=kx + 1, height=0.26, color="#c3c2b7", edgecolor=fs.SURFACE, linewidth=1.2, zorder=3)
    ax.text(kx, ky + 0.26, "polynomial", ha="left", va="bottom", fontsize=7.5, color=fs.INK2)
    ax.text(kx + 1.0, ky - 0.2, "path product", ha="left", va="top", fontsize=7.5, color=fs.INK2)
    ax.set_yticks(ytick)
    ax.set_yticklabels(ylab)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, available + 5.2)
    ax.spines["bottom"].set_bounds(0, available + 1)
    ax.set_xticks(range(0, available + 1))
    ax.set_ylim(-0.55, len(groups) - 0.3)
    ax.set_xlabel("multiplicative depth of the full circuit (levels)")
    handles = [Patch(color=fs.ORIG, label="original evaluator"), Patch(color=fs.FOLD, label="folding evaluator"),
               Patch(color=fs.CRIT_LT, label="levels beyond the chain")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.58, 1.0), handlelength=1.2,
               columnspacing=1.4, fontsize=8)
    savefig(fig, "f7_depth_feasibility", ev)


# ---------------------------------------------------------------------------------------------
# F8 -- Real-versus-simulated agreement / error, per degree (linear axis from zero: the two
# evaluators differ by <= 2e-6 on a scale where the quantity itself is ~7e-6).
# ---------------------------------------------------------------------------------------------
def fig_f8_real_vs_simulated() -> None:
    ev = Evidence("f8_real_vs_simulated")
    ev.get("m1_encrypted_multiseed.json", "/backend", note=f"REAL vs {SIMULATED}. {PROXY}. {K9_NOTE}.")
    n_per_seed = ev.get("m1_encrypted_multiseed.json", "/n_per_seed")
    n_seeds = ev.get("m1_encrypted_multiseed.json", "/seeds")

    fig, ax = plt.subplots(figsize=(fs.W_COL, 2.9))
    fs.value_grid(ax)
    ax.set_ylim(0, 10)
    scale = 1e6
    agree_all = []
    for artifact, color, marker, label, shift in (("m1_encrypted_multiseed.json", fs.ORIG, "o", "original evaluator", -0.13),
                                                  (FOLDED_ARTIFACT, fs.FOLD, "s", "coefficient-folding evaluator", 0.13)):
        degrees = ev.get(artifact, "/k_executed")
        mae_mean, mae_lo, mae_hi = [], [], []
        for degree in degrees:
            key = f"k={degree}"
            agree_all.append(ev.get(artifact, f"/pooled_across_seeds/{key}/agreement_real_vs_simulated/mean"))
            mae_mean.append(ev.get(artifact, f"/pooled_across_seeds/{key}/normalized_mae_real_vs_simulated/mean") * scale)
            mae_lo.append(ev.get(artifact, f"/pooled_across_seeds/{key}/normalized_mae_real_vs_simulated/min") * scale)
            mae_hi.append(ev.get(artifact, f"/pooled_across_seeds/{key}/normalized_mae_real_vs_simulated/max") * scale)
        err = [[m - lo for m, lo in zip(mae_mean, mae_lo)], [hi - m for m, hi in zip(mae_mean, mae_hi)]]
        xs = [d + shift for d in degrees]
        ax.plot(xs, mae_mean, color=color, linewidth=1.3, zorder=3)
        _errbars(ax, xs, mae_mean, err, color)
        ax.plot(xs, mae_mean, marker, color=color, label=label, **fs.ringed())
    ax.set_xlabel("comparison-polynomial degree $k$")
    ax.set_ylabel(r"normalized MAE, real vs. simulated ($\times10^{-6}$)")
    ax.set_xticks([3, 5, 7, 9])
    ax.set_xlim(2.3, 9.7)
    ax.set_ylim(0, 10)
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(loc="lower left", handlelength=1.6, borderaxespad=0.3)
    if min(agree_all) == 1.0:
        ax.text(9.6, 9.35, "margin signs agree in every evaluation", ha="right", va="center", fontsize=7.5, color=fs.MUTED)
    fig.tight_layout()
    savefig(fig, "f8_real_vs_simulated", ev)



# ---------------------------------------------------------------------------------------------
# F9 -- Continuous-feature task: fidelity floor of the approximate comparison against the exact comparison on
# quantized inputs (added analysis; test split, five seeds)
# ---------------------------------------------------------------------------------------------
def fig_f9_continuous_fidelity() -> None:
    ev = Evidence("f9_continuous_fidelity")
    art = "m1c_final_summary.json"
    ev.get(art, "/source", note=f"{SIMULATED}. {PROXY}. Added analysis (continuous-feature task), test split, five seeds. "
                               "The exact points at k=9 and k=15 are the quantized models; k=15 is simulation only.")
    degrees = [1, 3, 5, 7, 9]
    d_m, d_sd, l_m, l_sd = [], [], [], []
    for d in degrees:
        key = f"k={d}"
        d_m.append(ev.get(art, f"/floor/{key}/f1/differing_per_1000/mean"))
        d_sd.append(ev.get(art, f"/floor/{key}/f1/differing_per_1000/sd"))
        l_m.append(100.0 * ev.get(art, f"/floor/{key}/f1/lost_share_of_plain_positives/mean"))
        l_sd.append(100.0 * ev.get(art, f"/floor/{key}/f1/lost_share_of_plain_positives/sd"))
    ex5_d = ev.get(art, "/exact_B5/f1/differing_per_1000/mean")
    ex5_l = 100.0 * ev.get(art, "/exact_B5/f1/lost_share_of_plain_positives/mean")
    ex8_d = ev.get(art, "/exact_B8/f1/differing_per_1000/mean")
    ex8_l = 100.0 * ev.get(art, "/exact_B8/f1/lost_share_of_plain_positives/mean")

    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharex=True)
    for ax, ys, sds, ex5, ex8, ylab in (
        (axes[0], d_m, d_sd, ex5_d, ex8_d, "decisions differing from plaintext\n(per 1,000 stays)"),
        (axes[1], l_m, l_sd, ex5_l, ex8_l, "plaintext positives lost (%)"),
    ):
        fs.value_grid(ax)
        ax.plot(degrees, ys, color=fs.ORIG, linewidth=1.6, zorder=3)
        _errbars(ax, degrees, ys, [sds, sds], fs.ORIG)
        ax.plot(degrees, ys, "o", color=fs.ORIG, **fs.ringed())
        ax.plot([9], [ex5], "D", color=fs.INK, markersize=7, zorder=5)
        ax.plot([15], [ex8], "D", color=fs.INK, mfc=fs.SURFACE, mec=fs.INK, mew=1.4, markersize=7, zorder=5)
        ax.set_xlim(0.2, 17.6)
        ax.set_ylim(bottom=-0.04 * max(ys) * 1.3)
        ax.set_xticks([1, 3, 5, 7, 9, 15])
        ax.set_xlabel("comparison-polynomial degree $k$")
        ax.set_ylabel(ylab)
    top = max(d_m)
    axes[0].text(9.5, top * 0.07, "exact, 5 bins\n(real ciphertexts)", ha="left", va="bottom", fontsize=8, color=fs.INK2)
    axes[0].text(15, top * 0.30, "exact, 8 bins\n(simulated)", ha="center", va="bottom", fontsize=8, color=fs.INK2)
    axes[0].text(3.6, top * 0.72, "approximate comparison,\nfull-precision inputs", ha="left", va="center", fontsize=8, color=fs.INK2)
    fig.tight_layout()
    savefig(fig, "f9_continuous_fidelity", ev)


def fig_f10_thermometer_tradeoff() -> None:
    ev = Evidence("f10_thermometer_tradeoff")
    art = "m1c_thermo_final_summary.json"
    ev.get(art, "/source", note=f"REAL ciphertexts (cost panels) and plaintext models (utility panel). {PROXY}. Added study (structure- and threshold-hiding thermometer protocol), test split, five seeds; "
                               "four-level chain except the interpolation point, which runs on the locked chain.")
    Bs = [5, 8, 16]
    circuits = [("broadcast_thermometer_4level", "thermometer, broadcast", fs.ORIG, "o"),
                ("packed_thermometer_4level", "thermometer, packed", fs.FOLD, "s"),
                ("client_indicator_baseline_4level", "client knows structure and thresholds", fs.MUTED, "^")]
    up = {c: [ev.get(art, f"/by_B/B={B}/real/{c}/client_upload_mb") for B in Bs] for c, *_ in circuits}
    sec = {c: [ev.get(art, f"/by_B/B={B}/real/{c}/seconds_per_stay/mean") for B in Bs] for c, *_ in circuits}
    sec_sd = {c: [ev.get(art, f"/by_B/B={B}/real/{c}/seconds_per_stay/sd") for B in Bs] for c, *_ in circuits}
    i_up = ev.get(art, "/by_B/B=5/real/exact_interpolation_7level/client_upload_mb")
    i_sec = ev.get(art, "/by_B/B=5/real/exact_interpolation_7level/seconds_per_stay/mean")
    cost = [ev.get(art, f"/by_B/B={B}/quantized_plain/auroc_cost_vs_full_precision/mean") for B in Bs]
    cost_sd = [ev.get(art, f"/by_B/B={B}/quantized_plain/auroc_cost_vs_full_precision/sd") for B in Bs]

    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.55))
    for ax in axes:
        fs.value_grid(ax)
        ax.set_xticks(Bs)
        ax.set_xlabel("bins per input $B$")
    ax = axes[0]
    for c, lab, col, mk in circuits:
        ax.plot(Bs, up[c], marker=mk, color=col, linewidth=1.5, markersize=5, label=lab, zorder=3)
    ax.plot([5], [i_up], "D", color=fs.INK, markersize=6, zorder=5, label="interpolation (client knows thresholds)")
    ax.set_yscale("log")
    ax.set_ylabel("upload per stay (MB)")
    ax = axes[1]
    for c, lab, col, mk in circuits:
        ax.errorbar(Bs, sec[c], yerr=sec_sd[c], marker=mk, color=col, linewidth=1.5, markersize=5, capsize=2, zorder=3)
    ax.plot([5], [i_sec], "D", color=fs.INK, markersize=6, zorder=5)
    ax.set_ylabel("seconds per stay")
    ax = axes[2]
    ax.errorbar(Bs, cost, yerr=cost_sd, marker="o", color=fs.INK, linewidth=1.5, markersize=5, capsize=2, zorder=3)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("AUROC cost of quantization")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    savefig(fig, "f10_thermometer_tradeoff", ev)


def fig_graphical_abstract() -> None:
    """Graphical abstract for the Journal of Biomedical Informatics (separate file; at least 1328 px wide, longest side horizontal).
    Every number comes through the Evidence accessor; the text states which backend each comes from."""
    ev = Evidence("graphical_abstract")
    ctr = "m1c_final_summary.json"
    art = "m1c_thermo_final_summary.json"
    ev.get(ctr, "/floor/k=9/f1/differing_per_1000/mean", note=f"SIMULATED approximate comparison, degree 9, F1 threshold. {PROXY}.")
    floor_diff = ev.get(ctr, "/floor/k=9/f1/differing_per_1000/mean", fmt=".1f")
    floor_lost = ev.get(ctr, "/floor/k=9/f1/lost_share_of_plain_positives/mean", fmt=".1%")
    sim_total = 0
    for B in (5, 8, 16):
        for key in ("n_differing_f1_op_per_seed", "n_differing_sens80_op_per_seed"):
            sim_total += ev.reduce("sum", art, f"/by_B/B={B}/thermometer_simulated/{key}/*", note="SIMULATED differing decisions, summed over seeds")
    cost5 = ev.get(art, "/by_B/B=5/quantized_plain/auroc_cost_vs_full_precision/mean", fmt=".4f")
    cost16 = ev.get(art, "/by_B/B=16/quantized_plain/auroc_cost_vs_full_precision/mean", fmt=".4f")
    base = ev.get(art, "/by_B/B=5/real/client_indicator_baseline_4level/seconds_per_stay/mean")
    pack = ev.get(art, "/by_B/B=5/real/packed_thermometer_4level/seconds_per_stay/mean")
    interp = ev.get(art, "/by_B/B=5/real/exact_interpolation_7level/seconds_per_stay/mean")
    lock = ev.get(art, "/locked_chain_arm/by_B/B=5/packed_thermometer_locked_chain/mean_seconds_per_stay")

    fig = plt.figure(figsize=(6.6, 2.9))
    fig.patch.set_facecolor(fs.SURFACE)
    fig.text(0.5, 0.955, "Encrypted gradient-boosted trees: what the node comparison costs", ha="center", va="center", fontsize=10.5, color=fs.INK, fontweight="bold")
    cards = [(0.02, "1  Approximate comparison"), (0.345, "2  Exact comparison, quantized inputs"), (0.67, "3  Thresholds hidden, honest client")]
    bg = fig.add_axes([0, 0, 1, 1], zorder=0)
    bg.axis("off")
    for x0, title in cards:
        bg.add_patch(plt.Rectangle((x0, 0.17), 0.31, 0.71, transform=bg.transAxes, facecolor="#f3f2ef", edgecolor=fs.MUTED, linewidth=0.7))
        fig.text(x0 + 0.155, 0.845, title, ha="center", va="center", fontsize=8.2, color=fs.INK, fontweight="bold")
    # card 1
    fig.text(0.175, 0.665, floor_diff, ha="center", va="center", fontsize=26, color=fs.CRIT)
    fig.text(0.175, 0.545, "of 1,000 decisions differ\nat degree 9 (simulated)", ha="center", va="center", fontsize=8, color=fs.INK)
    fig.text(0.175, 0.37, f"{floor_lost} of the plaintext positives lost;\nthe ranking is largely kept;\ndegree 63 did not remove it", ha="center", va="center", fontsize=7.4, color=fs.INK2)
    # card 2
    fig.text(0.5, 0.665, str(int(sim_total)), ha="center", va="center", fontsize=26, color=fs.ORIG)
    fig.text(0.5, 0.545, "differing decisions at 5, 8 and 16 bins\n(simulated, thresholds between margins)", ha="center", va="center", fontsize=8, color=fs.INK)
    fig.text(0.5, 0.37, f"cost: AUROC lower by {cost5} to {cost16};\nexact for the quantized model only;\nthe first run had ties (rounding)", ha="center", va="center", fontsize=7.4, color=fs.INK2)
    # card 3: seconds per stay at 5 bins (real ciphertexts)
    ax = fig.add_axes([0.815, 0.37, 0.155, 0.31], zorder=3)
    ax.patch.set_alpha(0)
    labels = ["client knows\nthe thresholds", "thermometer,\npacked, 4 levels", "interpolation,\n7 levels", "thermometer,\npacked, 7 levels"]
    vals = [base, pack, interp, lock]
    cols = [fs.MUTED, fs.FOLD, fs.INK2, fs.FOLD_LT]
    ax.barh(range(4), vals, color=cols, height=0.62, zorder=3)
    ax.set_yticks(range(4))
    ax.set_yticklabels(labels, fontsize=6.0)
    ax.invert_yaxis()
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_xlabel("seconds per stay (real)", fontsize=6.5)
    fs.value_grid(ax) if hasattr(fs, "value_grid") else None
    for i, v in enumerate(vals):
        ax.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=6.5, color=fs.INK)
    ax.set_xlim(0, max(vals) * 1.25)
    fig.text(0.825, 0.215, "on one chain the three circuits trade\ntime against upload", ha="center", va="center", fontsize=6.6, color=fs.INK2)
    fig.text(0.5, 0.075, "Proxy cohort (MIMIC-IV-ED-derived), 5 training seeds; no clinical performance is claimed.", ha="center", va="center", fontsize=7.2, color=fs.INK2)
    tiff = os.path.join(OUT_DIR, "graphical_abstract.tiff")
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(tiff, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    print(f"wrote {tiff}")
    savefig(fig, "graphical_abstract", ev)


def main() -> int:
    fig_f3_delta_f1_vs_degree()
    fig_f4_real_latency_vs_degree()
    fig_f5_tree_count_scaling()
    fig_f6_tree_depth_scaling()
    fig_f7_depth_feasibility()
    fig_f8_real_vs_simulated()
    if os.path.exists(os.path.join(_HERE, "..", "results", "m1c_final_summary.json")):
        fig_f9_continuous_fidelity()
    if os.path.exists(os.path.join(_HERE, "..", "results", "m1c_thermo_final_summary.json")):
        fig_f10_thermometer_tradeoff()
        fig_graphical_abstract()
    print("Note: F1 (pipeline) and F2 (packing architecture) are conceptual diagrams built "
          "separately via TikZ in paper_assets/figures_src/ (see src/build_diagrams.py), not "
          "produced by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
