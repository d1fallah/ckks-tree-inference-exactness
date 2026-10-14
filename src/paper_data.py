"""
M1 -- Shared evidence accessor for every number that reaches a paper table, figure, or macro.

Per the supervisor's Q1 Paper Finalization Guide Sec 6 ("the manuscript must never contain a
number that cannot be traced to a committed artifact") and CLAUDE.md's Golden Rule, this module
is the ONLY place a results/*.json file is read for paper-asset generation. src/generate_latex_tables.py,
src/generate_figures.py and src/build_diagrams.py all go through the `Evidence` class below, so:

  - a number is read from a JSON in exactly one place (no second, silently-diverging derivation
    like the one this module replaces in table_plaintext_baseline -- see the closure log);
  - every emitted value carries its own provenance (artifact file + JSON pointer), written
    automatically as a sidecar next to the asset that used it;
  - src/verify_paper_assets.py can re-read the RAW json, independently replay the pointer or the
    reduction, and assert it still matches the FORMATTED STRING actually printed into the .tex --
    catching drift that a "provider returns a float" design cannot catch (a hand-edited or stale
    .tex still contains the old formatted string, not the old float).

Pointer syntax: a '/'-delimited path into the parsed JSON, e.g.
  "/pooled_across_seeds/k=5/latency_wall_clock_s_mean/mean"
Existing result-JSON keys contain '=' but never '/', so this is unambiguous. A path segment of
'*' is a wildcard, used only by `reduce()` and `series()` to select every matching child.

Four kinds of number, four methods, deliberately not fewer:
  get(...)      -- one value at one pointer in one artifact (the common case).
  reduce(...)   -- an aggregate (mean/max/min/len) over a wildcard pointer, e.g. a mean across
                   25 models. Records the reduction NAME and the wildcard, so the verifier can
                   replay the aggregation itself rather than trust it.
  series(...)   -- an ordered list of values for a figure's plotted series (no formatting).
  external(...) -- a value that does NOT come from this repository's artifacts: a number quoted
                   from a cited paper for the T9 comparison table. Requires a citation and a
                   locus (e.g. "Table 3, p. 7") -- the Golden Rule is satisfied by the citation,
                   not by a JSON file, and the verifier enforces that every external() call
                   carries both.
  literal(...)  -- a value that is a declared constant, not a measurement (a column header count,
                   a LaTeX spacing literal like "[2pt]"). Requires a `reason` so it is visibly a
                   deliberate exemption from the token scan, not an accidental unsourced number.

Naming note (Gate M1-N): no function parameter or local here is bound to a bare `k`, `N`, or
`lam` -- src/check_notation.py treats those as violations. Use `degree`, `ring_degree`, `sec_level`.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "..", "results")
PROVENANCE_DIR = os.path.join(_HERE, "..", "paper_assets", "provenance")

# Artifacts superseded by a later, differently-scoped result. Real (still gitignored-free,
# committed) files, kept for the closure-log record -- but no paper asset may source them.
# See docs/e2e_reproduction_report.md and M1_CLOSURE_LOG.md entries 6/10/12 for why each one
# was superseded and by what.
SUPERSEDED_ARTIFACTS = {
    "m1_ckks_sweep_simulated_repro.json": "reproduction check only; use m1_ckks_sweep_simulated.json",
    "m1_k_selection_v1_singleseed_latency.json": "single-seed (n=30, one model) latency; "
        "use m1_k_selection.json (5-seed real-CKKS latency)",
    "m1_encrypted_subsample.json": "single-seed (n=30, one model) real-execution arm; "
        "use m1_encrypted_multiseed.json (5 seeds x 25 patients)",
    "m1_scaling_ablation.json": "single-seed scaling pilot whose headline cell did not "
        "replicate across seeds; use m1_scaling_multiseed.json",
}

_JSON_CACHE: dict[str, Any] = {}


def _load_json(artifact: str) -> Any:
    if artifact in SUPERSEDED_ARTIFACTS:
        raise ValueError(
            f"{artifact} is a SUPERSEDED artifact ({SUPERSEDED_ARTIFACTS[artifact]}); "
            f"no paper asset may source it (src/paper_data.py SUPERSEDED_ARTIFACTS)"
        )
    if artifact not in _JSON_CACHE:
        path = os.path.join(RESULTS_DIR, artifact)
        with open(path, "r", encoding="utf-8") as f:
            _JSON_CACHE[artifact] = json.load(f)
    return _JSON_CACHE[artifact]


def resolve_pointer(data: Any, pointer: str) -> Any:
    """Walk a '/'-delimited pointer into a parsed JSON structure. Integer-looking segments index
    lists; everything else is a dict key. No wildcard support here -- see resolve_wildcard()."""
    node = data
    for seg in pointer.strip("/").split("/"):
        if isinstance(node, list):
            node = node[int(seg)]
        else:
            node = node[seg]
    return node


def resolve_wildcard(data: Any, pointer: str) -> list[Any]:
    """Like resolve_pointer, but exactly one segment may be '*', matching every key (dict) or
    every element (list) at that position. Returns the list of matched leaf values."""
    segs = pointer.strip("/").split("/")
    assert segs.count("*") == 1, f"reduce()/series() pointer needs exactly one '*': {pointer!r}"
    star = segs.index("*")
    head, tail = segs[:star], segs[star + 1 :]
    node = data
    for seg in head:
        node = node[int(seg)] if isinstance(node, list) else node[seg]
    children = node if isinstance(node, list) else list(node.values())
    out = []
    for child in children:
        v = child
        for seg in tail:
            v = v[int(seg)] if isinstance(v, list) else v[seg]
        out.append(v)
    return out


def _fmt(value: Any, fmt: str | None) -> str:
    if fmt is None:
        return str(value)
    if fmt == "pct":
        return f"{value * 100:.1f}\\%"
    if fmt == "sci2":  # mantissa with two decimals and a power of ten, set in math mode for prose use
        mant, exp = f"{value:.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"
    return format(value, fmt)


@dataclass
class _Record:
    kind: str  # "value" | "reduce" | "series" | "external" | "literal"
    artifact: str | None
    pointer: str | None
    reduce_how: str | None
    raw: Any
    fmt: str | None
    formatted: str
    note: str
    citation: str | None = None
    locus: str | None = None
    reason: str | None = None


class Evidence:
    """One instance per emitted asset (one table, one figure, one macro file). Accumulates
    _Record entries as get()/reduce()/series()/external()/literal() are called, then
    write_provenance() dumps them as that asset's sidecar."""

    def __init__(self, asset_id: str):
        self.asset_id = asset_id
        self.records: list[_Record] = []

    # ---- the four accessor kinds -------------------------------------------------------
    def get(self, artifact: str, pointer: str, *, fmt: str | None = None, note: str = "") -> Any:
        raw = resolve_pointer(_load_json(artifact), pointer)
        formatted = _fmt(raw, fmt)
        self.records.append(_Record("value", artifact, pointer, None, raw, fmt, formatted, note))
        return formatted if fmt is not None else raw

    def reduce(self, how: str, artifact: str, wildcard_pointer: str, *, fmt: str | None = None, note: str = "") -> Any:
        values = resolve_wildcard(_load_json(artifact), wildcard_pointer)
        if how == "mean":
            raw = sum(values) / len(values)
        elif how == "max":
            raw = max(values)
        elif how == "min":
            raw = min(values)
        elif how == "len":
            raw = len(values)
        elif how == "sum":
            raw = sum(values)
        else:
            raise ValueError(f"unknown reduction {how!r}")
        formatted = _fmt(raw, fmt)
        self.records.append(_Record(f"reduce:{how}", artifact, wildcard_pointer, how, raw, fmt, formatted, note))
        return formatted if fmt is not None else raw

    def series(self, artifact: str, wildcard_pointer: str, *, note: str = "") -> list[Any]:
        values = resolve_wildcard(_load_json(artifact), wildcard_pointer)
        self.records.append(_Record("series", artifact, wildcard_pointer, None, values, None, str(values), note))
        return values

    def external(self, value: Any, *, citation: str, locus: str, fmt: str | None = None, note: str = "") -> str:
        """A number quoted from a cited external paper (T9 only). NOT a repository artifact --
        the Golden Rule is satisfied by (citation, locus), which the verifier requires non-empty."""
        assert citation and locus, "external() requires both citation and locus"
        formatted = _fmt(value, fmt) if value != "NR" else "NR"
        self.records.append(_Record("external", None, None, None, value, fmt, formatted, note, citation=citation, locus=locus))
        return formatted

    def literal(self, value: Any, *, reason: str) -> Any:
        """A declared, non-measured constant (a column count, a spacing literal). Requires a
        reason so an unsourced number is a visible, deliberate exemption, not an accident."""
        assert reason, "literal() requires a reason"
        formatted = str(value)
        self.records.append(_Record("literal", None, None, None, value, None, formatted, "", reason=reason))
        return value

    # ---- sidecar --------------------------------------------------------------------------
    def write_provenance(self) -> str:
        os.makedirs(PROVENANCE_DIR, exist_ok=True)
        path = os.path.join(PROVENANCE_DIR, f"{self.asset_id}.json")
        payload = {
            "asset_id": self.asset_id,
            "records": [
                {
                    "kind": r.kind,
                    "artifact": r.artifact,
                    "pointer": r.pointer,
                    "reduce_how": r.reduce_how,
                    "raw": r.raw,
                    "fmt": r.fmt,
                    "formatted": r.formatted,
                    "note": r.note,
                    "citation": r.citation,
                    "locus": r.locus,
                    "reason": r.reason,
                }
                for r in self.records
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        return path


def real_latency_by_degree(ev: Evidence, artifact: str = "m1_encrypted_multiseed.json") -> dict[int, tuple[float, float]]:
    """Real-ciphertext latency per k from a 5-seed arm: {degree: (mean over seeds, SD across
    seeds)}. The default artifact is the original-evaluator arm (k=9 is absent there: it does not fit
    the locked chain); m1_encrypted_multiseed_folded.json is the coefficient-folding arm, which
    includes k=9. The single shared copy of the helper that both generate_latex_tables.py and
    generate_figures.py used to keep separately."""
    n_seeds = ev.get(artifact, "/seeds", note="sanity: 5 seeds")
    n_per_seed = ev.get(artifact, "/n_per_seed", note="sanity: 25 patients/seed")
    assert len(n_seeds) == 5 and n_per_seed == 25
    degrees = ev.get(artifact, "/k_executed", note="degrees with real execution")
    out = {}
    for deg in degrees:
        key = f"k={deg}"
        mean = ev.get(artifact, f"/pooled_across_seeds/{key}/latency_wall_clock_s_mean/mean", fmt=".3f")
        sd = ev.get(artifact, f"/pooled_across_seeds/{key}/latency_wall_clock_s_mean/sd_across_seeds", fmt=".3f")
        out[deg] = (float(mean), float(sd))
    return out


def emit_macro_file(macros: list[tuple], out_path: str) -> tuple[str, "Evidence"]:
    r"""Write paper_assets/tables/m1_numbers.tex: one \newcommand per EXPLICITLY NAMED prose/diagram
    quantity, for numbers quoted in the manuscript or the TikZ diagrams (F1/F2) rather than inside a
    generated table. This replaced an earlier index-based naming scheme (asset-id + record position)
    that silently rebinds every later macro's name whenever an earlier ev.get() call is added to the
    same function -- unusable once real prose starts citing specific macro names.

    `macros` is a list of (macro_suffix, kind, artifact, pointer, fmt, note) tuples, where `kind`
    is either "get" (one value at one pointer) or a reduce() method name ("max"/"min"/"mean"/"len",
    applied over a wildcard pointer), e.g.:
        ("RealLatencyDegThree", "get", "m1_encrypted_multiseed.json",
         "/pooled_across_seeds/k=3/latency_wall_clock_s_mean/mean", ".3f", "REAL ciphertext execution")
        ("MaxLeaves", "max", "m1_circuit_params.json",
         "/split_census/per_model/*/n_leaves", ",", "max over 25 frozen models")
    emitting \newcommand{\MOneRealLatencyDegThree}{0.658}. macro_suffix must be a fixed, stable name
    chosen by the caller -- never derived from position -- so a manuscript citation of \MOneFoo keeps
    meaning "Foo" even after unrelated macros are added or removed.

    Returns (out_path, ev): the same Evidence instance used, so the caller can also call
    ev.write_provenance() and get this file's own sidecar (paper_assets/provenance/m1_numbers.json),
    covered by verify_paper_assets.py's existing checks with no changes to that script.

    Naming (Gate M1-N): macro_suffix must spell degrees out (DegThree, not KThree) -- src/check_notation.py
    flags bare k/N/lam bindings in Python, and by convention here bare 'K' is avoided in macro names too
    so a manuscript search for the locked symbol does not collide with macro names.
    """
    ev = Evidence("m1_numbers")
    seen: set[str] = set()
    header = [
        "% AUTO-GENERATED by src/paper_data.py:emit_macro_file -- do not hand-edit.",
        "% Each \\newcommand name is an explicit, stable identifier chosen at the call site in",
        "% generate_paper_macros.py -- never derived from position, so it cannot silently rebind.",
        "",
    ]
    lines = list(header)
    for suffix, kind, artifact, pointer, fmt, note in macros:
        assert suffix not in seen, f"duplicate macro suffix {suffix!r}"
        seen.add(suffix)
        # A LaTeX control WORD is a backslash followed by a MAXIMAL RUN OF LETTERS ONLY -- TeX
        # stops reading the name at the first non-letter, including a digit. \newcommand{\Foo1Bar}
        # is therefore NOT "define \Foo1Bar"; it is "define \Foo" with the literal text "1Bar"
        # left over inside the same brace group, which \newcommand's argument parser then chokes
        # on -- reliably, but only at LaTeX COMPILE time, with an opaque "Missing \begin{document}"
        # error far from this file. Found empirically (macro suffix "MacroF1Mean" broke a stub
        # compile); every future suffix is checked here instead of being re-discovered the same way.
        assert suffix.isalpha(), (
            f"macro suffix {suffix!r} contains a non-letter (likely a digit, e.g. from 'F1') -- "
            f"LaTeX control words cannot contain digits; spell it out (e.g. 'FOne') in generate_paper_macros.py"
        )
        name = "MOne" + suffix
        if kind == "get":
            value = ev.get(artifact, pointer, fmt=fmt, note=note)
        else:
            value = ev.reduce(kind, artifact, pointer, fmt=fmt, note=note)
        # A bare '%' is a LaTeX comment marker -- Evidence._fmt's "pct" spec already escapes it,
        # but the "%" Python-format-spec (used for AtcCoverage etc.) does not, so escape any
        # generically here rather than relying on every fmt spec to have remembered to.
        val_tex = str(value).replace("%", "\\%").replace("_", "\\_")
        lines.append("\\newcommand{\\" + name + "}{" + val_tex + "}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    newline = chr(10)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(newline.join(lines) + newline)
    return out_path, ev
