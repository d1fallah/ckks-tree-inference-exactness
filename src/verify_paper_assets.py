"""
M1 -- Independent verification gate for every paper table, figure and macro (Q1 Paper
Finalization Guide Sec 31's QC checklist, made mechanical).

CRITICAL DESIGN RULE: this script never imports src/paper_data.py's Evidence accessors for
verification. If it did, every check would collapse to "the provider agrees with itself",
which passes forever even if a committed .tex is stale or was hand-edited. Instead it:

  1. Re-reads the RAW results/*.json files directly (never through Evidence.get/reduce).
  2. Re-reads each emitted asset's provenance sidecar (paper_assets/provenance/<asset>.json),
     which records (artifact, pointer, reduce_how, raw, fmt, formatted) for every number that
     asset used.
  3. REPLAYS each sidecar record independently: re-resolve the pointer (or re-run the named
     reduction) against the raw JSON, and re-apply the format spec -- then compares both against
     what the sidecar (and, for check 4, the actual committed .tex bytes) say.

Checks performed, in order (numbered to match the approved plan):
  1. Manifest integrity      -- every artifact any sidecar names is in results/m1_evidence_manifest.json,
                                 and its on-disk SHA-256 matches the manifest's.
  2. Replay                  -- resolving/reducing the raw JSON again gives the same value the
                                 sidecar recorded (catches a sidecar written against a JSON that
                                 has since changed without a re-run).
  3. Re-format                -- format(raw, fmt) == formatted (catches a changed fmt spec that
                                 was never re-run, or a hand-typed formatted string).
  4. Token scan               -- every numeric literal actually printed in each paper_assets/tables/*.tex
                                 is either one of that asset's own recorded `formatted` values or on
                                 its declared literal() whitelist (catches a stale or hand-edited .tex:
                                 the sidecar can be perfectly self-consistent while the committed file
                                 next to it is not what it claims to have produced).
  5. Cross-asset agreement    -- every (artifact, pointer) pair, grouped across ALL sidecars, must
                                 carry the same raw value everywhere it appears (mechanizes "every
                                 figure agrees with the tables" without a separate CSV).
  6. QC label rules           -- SIMULATED / REAL / Proxy / "simulation only" / "additional arm"
                                 wording rules from guide Sec 31, checked over sidecar `note` fields
                                 (the caption text a table/figure actually carries).
  7. Document scan            -- every literal with >= 3 decimal digits or scientific notation, in
                                 MANUSCRIPT-FACING documents only (M1_FINAL_EVIDENCE_FREEZE.md,
                                 README.md, and Phase-5 methods/results/protocol docs -- see
                                 MANUSCRIPT_FACING_DOC_* below for the exact scope and why the
                                 pre-existing closure-phase audit documents are excluded), must
                                 appear in SOME sidecar's formatted-value set. Filters out years,
                                 seeds, section numbers by the decimal-length threshold, and DOI/
                                 arXiv identifiers by citation context.
  8. External citations       -- every external() record (a number quoted from a cited paper, T9
                                 only) carries a non-empty citation and locus, the citation is a
                                 bibkey in paper/refs/m1.bib once that file exists, and an "NR" raw
                                 value always renders as the literal string "NR" (guide Sec 26).

Also refuses (fails loudly, does not silently skip) if any sidecar names a SUPERSEDED_ARTIFACTS entry.

Exit 0: all checks pass. Exit 1: grouped, explicit failure listing.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
from paper_data import SUPERSEDED_ARTIFACTS, resolve_pointer, resolve_wildcard  # noqa: E402

PROVENANCE_DIR = os.path.join(ROOT, "paper_assets", "provenance")
TABLES_DIR = os.path.join(ROOT, "paper_assets", "tables")
FLOATS_DIR = os.path.join(ROOT, "paper_draft", "floats")
RESULTS_DIR = os.path.join(ROOT, "results")
MANIFEST_PATH = os.path.join(RESULTS_DIR, "m1_evidence_manifest.json")

# Numbers below 1e-9 in absolute terms are treated as equal to avoid float noise from json
# round-tripping; this is replay tolerance, not measurement tolerance.
REPLAY_ATOL = 1e-9

TOKEN_RE = re.compile(r"(?<![\w.])-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?(?![\w.])")
# Word-boundary-guarded so this does not match inside hex color codes / identifiers like "f59e0b"
# (which contains "59e0", a spurious "scientific notation" match without the boundary) or commit
# hashes. (?<![\w.]) / (?![\w.]) rather than \b: \b alone still allows a match starting right
# after another digit run with no separator, which a bare hex/identifier string is.
DOC_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+\.\d{3,}(?![\w.])|(?<![\w.])-?\d+(?:\.\d+)?[eE][+-]?\d+(?![\w.])")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest() -> dict:
    if not os.path.exists(MANIFEST_PATH):
        raise SystemExit(f"no evidence manifest at {MANIFEST_PATH} -- run src/freeze_evidence.py first")
    return load_json(MANIFEST_PATH)


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_sidecars() -> dict[str, dict]:
    out = {}
    for path in sorted(glob.glob(os.path.join(PROVENANCE_DIR, "*.json"))):
        d = load_json(path)
        out[d["asset_id"]] = d
    return out


def _fmt(value, fmt):
    if fmt is None:
        return str(value)
    if fmt == "pct":
        return f"{value * 100:.1f}\\%"
    if fmt == "sci2":  # mantissa with two decimals and a power of ten, set in math mode for prose use
        mant, exp = f"{value:.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"
    return format(value, fmt)


def _numeric_like_simple(a: str) -> bool:
    """True if `a` is (up to a leading '-') a plain number with at most one '.': the kind of
    string a bare formatted numeric VALUE produces, as opposed to a longer prose/descriptive
    literal() string that happens to contain digits."""
    body = a[1:] if a.startswith("-") else a
    return body != "" and body.count(".") <= 1 and body.replace(".", "").replace(",", "").isdigit()


def _close(a, b) -> bool:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= REPLAY_ATOL
    return a == b


def check_manifest_integrity(sidecars: dict, manifest: dict, failures: list) -> None:
    manifest_files = manifest.get("files", {})
    for asset_id, sc in sidecars.items():
        for rec in sc["records"]:
            artifact = rec.get("artifact")
            if artifact is None:
                continue
            key = f"results/{artifact}"
            if key not in manifest_files:
                failures.append(f"[manifest] {asset_id}: artifact {artifact!r} is not in the evidence manifest "
                               f"(run src/freeze_evidence.py after adding a new source artifact)")
                continue
            on_disk = os.path.join(RESULTS_DIR, artifact)
            if not os.path.exists(on_disk):
                failures.append(f"[manifest] {asset_id}: artifact {artifact!r} listed in the manifest but missing on disk")
                continue
            actual_hash = sha256_file(on_disk)
            if actual_hash != manifest_files[key]:
                failures.append(f"[manifest] {asset_id}: {artifact!r} hash on disk does not match the frozen manifest "
                               f"-- the evidence freeze is stale (run freeze_evidence.py --amend)")


def check_replay_and_format(sidecars: dict, failures: list) -> None:
    json_cache: dict[str, dict] = {}
    for asset_id, sc in sidecars.items():
        for i, rec in enumerate(sc["records"]):
            if rec["kind"] in ("external", "literal"):
                continue
            artifact = rec["artifact"]
            if artifact not in json_cache:
                path = os.path.join(RESULTS_DIR, artifact)
                if not os.path.exists(path):
                    failures.append(f"[replay] {asset_id}#{i}: artifact {artifact!r} not found on disk")
                    continue
                json_cache[artifact] = load_json(path)
            data = json_cache[artifact]
            try:
                if rec["kind"] == "value":
                    replayed = resolve_pointer(data, rec["pointer"])
                elif rec["kind"].startswith("reduce:"):
                    values = resolve_wildcard(data, rec["pointer"])
                    how = rec["reduce_how"]
                    replayed = {"mean": lambda v: sum(v) / len(v), "max": max, "min": min, "len": len, "sum": sum}[how](values)
                elif rec["kind"] == "series":
                    replayed = resolve_wildcard(data, rec["pointer"])
                else:
                    continue
            except (KeyError, IndexError, ZeroDivisionError) as exc:
                failures.append(f"[replay] {asset_id}#{i}: could not re-resolve {rec['pointer']!r} in {artifact!r}: {exc}")
                continue
            if not _close(replayed, rec["raw"]):
                failures.append(f"[replay] {asset_id}#{i}: {artifact}:{rec['pointer']} re-resolves to {replayed!r}, "
                               f"sidecar recorded {rec['raw']!r} -- the source artifact changed since generation")
                continue
            if rec["kind"] != "series":
                refmt = _fmt(replayed, rec["fmt"])
                if refmt != rec["formatted"]:
                    failures.append(f"[format] {asset_id}#{i}: re-formatting {replayed!r} with {rec['fmt']!r} gives "
                                   f"{refmt!r}, sidecar says {rec['formatted']!r}")


def check_token_scan(sidecars: dict, failures: list) -> None:
    for asset_id, sc in sidecars.items():
        tex_path = os.path.join(TABLES_DIR, f"{asset_id}.tex")
        if not os.path.exists(tex_path):
            continue  # figures have no .tex to scan; token scan is a table-only check
        with open(tex_path, "r", encoding="utf-8") as f:
            tex = f.read()
        allowed = {rec["formatted"] for rec in sc["records"] if rec["kind"] != "series"}
        allowed |= {str(rec["raw"]) for rec in sc["records"] if rec["kind"] == "literal"}
        # A formatted value ending in a bare '%' (e.g. "83.9%") is written into a .tex file as
        # "83.9\%" (a bare '%' is a LaTeX comment marker -- see paper_data.emit_macro_file), so
        # TOKEN_RE only ever recovers "83.9" from the source text. Accept the percent-stripped
        # form too, rather than trying to make the token regex escape-aware.
        allowed |= {a.rstrip("%") for a in list(allowed) if a.endswith("%")}
        # Prose-length allowed strings (a caption fragment or a joined display like "42, 123, 7,
        # 456, 789" registered via literal()) can never be produced whole by TOKEN_RE, which
        # tokenizes individual numbers -- a caption containing one legitimately produces several
        # sub-matches that must be allowed by containment, not exact equality. Restricted to
        # values noticeably longer than a single number (>6 chars) so this cannot accidentally
        # make one short numeric value "explain" another by coincidence.
        prose_allowed = [a for a in allowed if not _numeric_like_simple(a) and (len(a) > 6 or a.count(".") >= 2)]

        tokens_found = set(TOKEN_RE.findall(tex))
        # Filter out small integers that are almost certainly structural (column widths in labels,
        # degree values k=1/3/5/7/9 already appear as formatted strings in practice) -- only flag
        # tokens with a decimal point or 3+ digits, which is where a stray/stale number would show up.
        suspects = {t for t in tokens_found if ("." in t or len(t.replace(",", "").lstrip("-")) >= 3)}

        def explained(token: str) -> bool:
            stripped = token.rstrip(".")  # a caption's sentence-ending period attaches to the last digit
            if token in allowed or stripped in allowed:
                return True
            bare = stripped.rstrip("0").rstrip(".")
            if any(bare == a.rstrip("0").rstrip(".") for a in allowed if "." in a):
                return True
            if any(stripped in a for a in prose_allowed):
                return True
            return False

        unexplained = {t for t in suspects if not explained(t)}
        if unexplained:
            failures.append(f"[token-scan] {asset_id}.tex contains numeric literal(s) not traced to any "
                           f"recorded value or literal(): {sorted(unexplained)}")


def check_cross_asset_agreement(sidecars: dict, failures: list) -> None:
    # Grouping key includes reduce_how: a wildcard pointer legitimately produces different raw
    # values under different reductions (e.g. min vs max over the same
    # "/per_configuration/*/memory_kb/..." series is two different, both-correct numbers) --
    # conflating them under (artifact, pointer) alone is a false positive, not a real disagreement.
    by_key: dict[tuple, list] = {}
    for asset_id, sc in sidecars.items():
        for rec in sc["records"]:
            if rec["kind"] in ("external", "literal") or rec["artifact"] is None:
                continue
            key = (rec["artifact"], rec["pointer"], rec.get("reduce_how"))
            by_key.setdefault(key, []).append((asset_id, rec["raw"]))
    for key, entries in by_key.items():
        values = [v for _, v in entries]
        first = values[0]
        if not all(_close(first, v) for v in values):
            failures.append(f"[cross-asset] {key[0]}:{key[1]} (reduce={key[2]}) disagrees across assets: "
                           f"{[(a, v) for a, v in entries]}")


QC_REQUIRED_SIMULATED_SOURCES = {"m1_ckks_sweep_simulated.json", "m1_ckks_sweep_k1_simulated.json"}
QC_FOLDED_REAL_SOURCES = {"m1_encrypted_multiseed_folded.json", "m1_folded_depth_accounting.json", "m1_latency_breakdown_folded.json"}


def _has_caption(tex_path: str) -> bool:
    with open(tex_path, "r", encoding="utf-8") as f:
        return "\\caption{" in f.read()


def check_qc_labels(sidecars: dict, failures: list) -> None:
    for asset_id, sc in sidecars.items():
        sources = {rec["artifact"] for rec in sc["records"] if rec.get("artifact")}
        notes = " ".join(rec.get("note", "") or "" for rec in sc["records"])
        # The wording rules govern what a READER sees, so check the actual committed caption/label
        # text -- a generated table (paper_assets/tables/{id}.tex) or, once Phase 3 wired floats
        # into the manuscript, a hand-authored figure caption (paper/floats/{id}.tex). A
        # \newcommand macro file like m1_numbers.tex has neither, so it falls back to sidecar notes.
        tex_path = os.path.join(TABLES_DIR, f"{asset_id}.tex")
        float_path = os.path.join(FLOATS_DIR, f"{asset_id}.tex")
        if os.path.exists(tex_path) and _has_caption(tex_path):
            with open(tex_path, "r", encoding="utf-8") as f:
                caption_text = f.read()
        elif os.path.exists(float_path) and _has_caption(float_path):
            with open(float_path, "r", encoding="utf-8") as f:
                caption_text = f.read()
        else:
            caption_text = notes

        # k=9 labelling (revised with the coefficient-folding evaluator): a k=9 value read from a SIMULATED artifact must say
        # "simulation only"; a k=9 value read from a REAL folding-evaluator artifact must name the evaluator ("folding").
        has_k9_row = any("k=9" in (rec.get("pointer") or "") and rec.get("artifact") in QC_REQUIRED_SIMULATED_SOURCES
                         for rec in sc["records"])
        has_k9_real_folded_row = any("k=9" in (rec.get("pointer") or "") and rec.get("artifact") in QC_FOLDED_REAL_SOURCES
                                     for rec in sc["records"])
        has_k1_row = any("k=1" in (rec.get("pointer") or "") or "m1_ckks_sweep_k1_simulated.json" == rec.get("artifact")
                          for rec in sc["records"])

        if sources & QC_REQUIRED_SIMULATED_SOURCES and "simulated" not in caption_text.lower():
            failures.append(f"[qc-label] {asset_id}: sources a simulated artifact but caption/note lacks 'SIMULATED'")
        if has_k9_row and "simulation only" not in caption_text and "no real" not in caption_text.lower():
            failures.append(f"[qc-label] {asset_id}: has a k=9 row but does not state simulation-only status")
        if has_k9_real_folded_row and "folding" not in caption_text.lower():
            failures.append(f"[qc-label] {asset_id}: has a real k=9 value but does not name the coefficient-folding evaluator")
        if "m1_encrypted_multiseed.json" in sources:
            for rec in sc["records"]:
                if rec.get("artifact") == "m1_encrypted_multiseed.json" and "k=9" in (rec.get("pointer") or ""):
                    failures.append(f"[qc-label] {asset_id}: sources m1_encrypted_multiseed.json at a k=9 pointer "
                                   f"-- k=9 has no real execution")
        if has_k1_row and not ("additional" in caption_text and "arm" in caption_text):
            failures.append(f"[qc-label] {asset_id}: has a k=1 row but does not label it an additional/diagnostic arm")
        if sources and "Proxy" not in caption_text:
            failures.append(f"[qc-label] {asset_id}: sources a results artifact but caption/note lacks 'Proxy' cohort labeling")
        for rec in sc["records"]:
            if rec.get("artifact") in SUPERSEDED_ARTIFACTS:
                failures.append(f"[qc-label] {asset_id}: sources SUPERSEDED artifact {rec['artifact']!r} "
                               f"({SUPERSEDED_ARTIFACTS[rec['artifact']]})")


# Documents whose job is to RESTATE numbers that live in paper-asset sidecars: the evidence
# freeze, the README, and every Phase-5 methods/results/protocol document (named by this fixed
# prefix convention so a newly-added Phase-5 doc is covered without editing this list).
#
# Deliberately NOT scanned: the pre-existing closure-phase audit/protocol documents
# (docs/anchor_reselection_preregistration.md, docs/e2e_reproduction_report.md,
# docs/error_decomposition.md, docs/framing_brief_for_supervisor.md,
# docs/gamma_selection_protocol.md, docs/guide_verification_note.md,
# docs/k1_compatibility.md, docs/leakage_and_threshold_audit.md, docs/security_estimate.md).
# Each of those is a PRIMARY SOURCE document in its own right -- it documents its own
# computation, from its own script, reviewed and committed during the M1 closure phase -- not a
# restatement of a paper-table number, so it has no reason to appear in any table's sidecar.
# Scanning them produced ~100 false positives in the first version of this check (every number
# in every closure-phase document), which is exactly the "noisy gate nobody reads" failure mode
# this mechanism exists to avoid. Only a document whose purpose is to quote a table/figure value
# belongs in this scan.
MANUSCRIPT_FACING_DOC_NAMES = {"M1_FINAL_EVIDENCE_FREEZE.md", "README.md"}
MANUSCRIPT_FACING_DOC_PREFIXES = ("docs/methods_", "docs/experimental_protocol", "M1_Results_Documentation")

CITATION_CONTEXT_RE = re.compile(r"\b(doi|arxiv|isbn|issn)\b", re.IGNORECASE)
# A number explicitly quoted as a past, corrected value (e.g. "an earlier version reported X,
# since retracted") is not a current claim and has no reason to trace to a live artifact -- it is
# the wrong number, kept on record for the correction to make sense. Exempted by nearby context.
RETRACTED_CONTEXT_RE = re.compile(r"\b(retracted|superseded|since\s+corrected|was\s+wrong)\b", re.IGNORECASE)


def _manuscript_facing_docs() -> list[str]:
    paths = []
    for name in MANUSCRIPT_FACING_DOC_NAMES:
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            paths.append(p)
    for p in sorted(glob.glob(os.path.join(ROOT, "docs", "*.md"))) + sorted(glob.glob(os.path.join(ROOT, "M1_*.md"))):
        rel = os.path.relpath(p, ROOT).replace(os.sep, "/")
        if any(rel.startswith(prefix) for prefix in MANUSCRIPT_FACING_DOC_PREFIXES):
            paths.append(p)
    # The manuscript's own prose (hand-authored .tex, unlike the generated paper_assets/tables/
    # and _src TikZ diagrams) was never added here despite paper/README.md's Phase-0 note saying
    # it would be -- found while wiring floats into the document for the first time. DOC_NUMBER_RE
    # only matches long-decimal/scientific-notation tokens (see its definition), so this is the
    # same narrow, low-noise net already used for the closure docs above, not the ~100-false-positive
    # broad scan that was rejected.
    paths += sorted(glob.glob(os.path.join(ROOT, "paper_draft", "sections", "*.tex")))
    paths += sorted(glob.glob(os.path.join(ROOT, "paper_draft", "floats", "*.tex")))
    return sorted(set(paths))


def check_external_citations(sidecars: dict, failures: list) -> None:
    """Enforce what paper_data.py's docstring already claims: every external() record (a number
    quoted from a cited paper, used only by T9) must carry a non-empty citation and locus -- the
    Golden Rule is satisfied by the citation, not by a results artifact, for exactly these cells.
    If paper/refs/m1.bib exists, also require `citation` to be a bibkey defined there, so a typo'd
    or unverified citation cannot silently sit in a published comparison table."""
    bib_keys: set[str] | None = None
    bib_path = os.path.join(ROOT, "paper", "refs", "m1.bib")
    if os.path.exists(bib_path):
        with open(bib_path, "r", encoding="utf-8") as f:
            bib_text = f.read()
        bib_keys = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", bib_text))

    for asset_id, sc in sidecars.items():
        for i, rec in enumerate(sc["records"]):
            if rec["kind"] != "external":
                continue
            if not rec.get("citation") or not rec.get("locus"):
                failures.append(f"[external-citation] {asset_id}#{i}: external() value missing a "
                               f"citation and/or locus -- both are required (paper_data.py docstring)")
            elif bib_keys is not None and rec["citation"] not in bib_keys:
                failures.append(f"[external-citation] {asset_id}#{i}: citation {rec['citation']!r} "
                               f"is not a bibkey in paper/refs/m1.bib")
            if rec["raw"] == "NR" and rec["formatted"] != "NR":
                failures.append(f"[external-citation] {asset_id}#{i}: raw value is 'NR' (not reported) "
                               f"but formatted as {rec['formatted']!r} -- must render as literal 'NR'")


def check_document_scan(sidecars: dict, failures: list) -> None:
    all_formatted: set[str] = set()
    for sc in sidecars.values():
        all_formatted |= {rec["formatted"] for rec in sc["records"] if rec["kind"] != "series"}
    all_formatted_bare = {v.replace("\\%", "").replace("$", "").strip() for v in all_formatted}

    for path in _manuscript_facing_docs():
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        lines = text.split("\n")
        for m in DOC_NUMBER_RE.finditer(text):
            token = m.group(0)
            line_no = text.count("\n", 0, m.start())
            if CITATION_CONTEXT_RE.search(lines[line_no]):
                continue  # DOIs / arXiv IDs are citations, not measurements
            window = "\n".join(lines[max(0, line_no - 1):line_no + 4])  # a retraction note may
            if RETRACTED_CONTEXT_RE.search(window):                     # span a few lines
                continue  # explicitly quoted as a past, corrected value -- not a current claim
            if token in all_formatted or token in all_formatted_bare:
                continue
            # tolerate a trailing-zero / rounding mismatch (e.g. doc says 0.658, asset stored 0.6580).
            if any(abs(float(token.replace(",", "")) - float(a.replace(",", ""))) < 1e-6
                   for a in all_formatted_bare if _numeric_like_simple(a)):
                continue
            failures.append(f"[doc-scan] {os.path.relpath(path, ROOT)}: literal {token!r} does not match any "
                           f"recorded paper-asset value -- verify it traces to a committed artifact")


def main() -> int:
    manifest = load_manifest()
    sidecars = load_sidecars()
    if not sidecars:
        print("NOTE: no provenance sidecars found under paper_assets/provenance/ -- "
              "run generate_latex_tables.py / generate_figures.py first. Nothing to verify.")
        return 0

    failures: list[str] = []
    check_manifest_integrity(sidecars, manifest, failures)
    check_replay_and_format(sidecars, failures)
    check_token_scan(sidecars, failures)
    check_cross_asset_agreement(sidecars, failures)
    check_qc_labels(sidecars, failures)
    check_external_citations(sidecars, failures)
    check_document_scan(sidecars, failures)

    if failures:
        print(f"--- FAIL: {len(failures)} paper-asset verification failure(s) ---")
        for f in failures:
            print(f"  {f}")
        return 1

    n_records = sum(len(sc["records"]) for sc in sidecars.values())
    print(f"paper-asset verification: PASS ({len(sidecars)} assets, {n_records} recorded values, "
          f"all replayed/reformatted/cross-checked against {len(manifest.get('files', {}))} manifested artifacts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
