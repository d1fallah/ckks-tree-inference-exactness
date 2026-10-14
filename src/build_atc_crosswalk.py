#!/usr/bin/env python3
"""
Build an offline NDC -> RxNorm -> ATC crosswalk for M1's medication features.

MIMIC-IV-ED's `ed/medrecon.csv` carries no ATC codes -- only free-text `name`,
`gsn` (Generic Sequence Number), `ndc` (National Drug Code), and MIMIC's own
`etccode`/`etcdescription` (Enhanced Therapeutic Class) hierarchy. The roadmap
requires hierarchical ATC embeddings, so this script maps each *distinct* NDC
to its ATC classification via the public NLM RxNav / RxClass REST API:

    NDC --[ndcstatus.json]--> RxCUI --[rxclass/class/byRxcui.json]--> ATC classes

Two live-verified request shapes (2026-09-11):
    GET https://rxnav.nlm.nih.gov/REST/ndcstatus.json?ndc={ndc}
        -> {"ndcStatus": {"rxcui": "...", "status": "ACTIVE"|"OBSOLETE"|"UNKNOWN"|..., ...}}
        status "UNKNOWN" with an empty rxcui means RxNorm has no record of
        this NDC at all -- MIMIC's own "0" sentinel (17,775 medrecon rows,
        meaning "no NDC recorded") resolves this way and is cached like any
        other unmapped value, not special-cased.
    GET https://rxnav.nlm.nih.gov/REST/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC
        -> {"rxclassDrugInfoList": {"rxclassDrugInfo": [
               {"rxclassMinConceptItem": {"classId": "R03AC", "className": "...", "classType": "ATC1-4"}}, ...
           ]}}
        RxClass's ATC coverage tops out at ATC level 4 (5-character classId,
        e.g. "R03AC") -- level 5 (the specific chemical substance, a 7-char
        code) is not returned. ATC levels 1-4 are derived from classId by
        substring (level 1 = classId[0], level 2 = classId[0:3],
        level 3 = classId[0:4], level 4 = classId[0:5]), giving a genuine
        hierarchical embedding without needing level 5.

Rate-limit discipline (NLM RxNav enforces ~20 req/s): only *distinct* NDCs are
queried (9,312 distinct real values across 2,987,342 medrecon rows, pandas
CSV-parsed count verified 2026-09-11 -- a ~321x reduction), each HTTP call
sleeps 50 ms afterward, and every response is written straight into the
on-disk JSON cache as it arrives so an interrupted run resumes from where it
left off instead of re-querying already-resolved NDCs.

Connection reuse: a single persistent HTTPS connection to rxnav.nlm.nih.gov is
kept open for the whole run (`http.client.HTTPSConnection`, reconnecting on
failure) rather than opening a fresh connection per request. Measured
2026-09-11: a cold connection (TLS handshake + RTT) costs ~0.5-0.8s per
request, while a request on an already-open connection costs ~0.15-0.2s --
a fresh-connection-per-request design would have made the full ~9,300-NDC
crosswalk take on the order of 7-8 hours; with reuse it is closer to 1-1.5
hours, which is what makes an offline, resumable cache practical here at all.

Usage:
    python build_atc_crosswalk.py [--limit N] [--cache-path PATH]
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mimic_paths import load_medrecon  # noqa: E402

RXNAV_HOST = "rxnav.nlm.nih.gov"
SLEEP_S = 0.05  # 50ms between HTTP calls, per pre-registered rate-limit discipline
MAX_RETRIES = 3
TIMEOUT_S = 10
FLUSH_EVERY = 25  # checkpoint the cache to disk every N newly-resolved NDCs
MIMIC_NULL_NDC = "0"  # MIMIC-IV-ED's own sentinel for "no NDC recorded"


class RxNavClient:
    """A single kept-alive HTTPS connection to RxNav, reconnecting on failure.

    Plain http.client rather than requests/urllib3 -- no extra dependency is
    needed for a single-host keep-alive GET client, and the WSL venv this
    script runs in intentionally carries no packages beyond what the Track A
    pipeline itself needs (pandas/numpy/xgboost/tenseal), per the isolation
    decision recorded in m1_vendor_provenance.md.
    """

    def __init__(self, host: str = RXNAV_HOST, timeout: float = TIMEOUT_S):
        self.host = host
        self.timeout = timeout
        self._conn: http.client.HTTPSConnection | None = None

    def _ensure_connected(self) -> None:
        if self._conn is None:
            self._conn = http.client.HTTPSConnection(self.host, timeout=self.timeout)

    def get_json(self, path: str) -> dict:
        last_exc: Exception | None = None
        for attempt in range(MAX_RETRIES):
            try:
                self._ensure_connected()
                self._conn.request("GET", path, headers={"User-Agent": "SihhaDZ-M1/1.0"})
                resp = self._conn.getresponse()
                body = resp.read().decode("utf-8")
                if resp.status != 200:
                    raise RuntimeError(f"HTTP {resp.status} for {path}")
                return json.loads(body)
            except Exception as exc:  # connection reset, timeout, bad status, bad JSON
                last_exc = exc
                try:
                    if self._conn is not None:
                        self._conn.close()
                except Exception:
                    pass
                self._conn = None  # force a fresh connection on the next attempt
                time.sleep(SLEEP_S * (2**attempt))
        raise RuntimeError(f"failed after {MAX_RETRIES} retries: {path}") from last_exc

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

DEFAULT_CACHE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "atc_crosswalk.json"
)
DEFAULT_COVERAGE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "m1_atc_crosswalk_coverage.json"
)


def resolve_ndc(client: RxNavClient, ndc: str) -> dict:
    """One NDC -> {rxcui, status, atc_classes: [{classId, className}]}."""
    status_path = f"/REST/ndcstatus.json?ndc={ndc}"
    status_resp = client.get_json(status_path)
    time.sleep(SLEEP_S)
    ndc_status = status_resp.get("ndcStatus", {}) or {}
    rxcui = ndc_status.get("rxcui") or ""
    status = ndc_status.get("status") or "UNKNOWN"

    atc_classes: list[dict] = []
    if rxcui:
        class_path = f"/REST/rxclass/class/byRxcui.json?rxcui={rxcui}&relaSource=ATC"
        class_resp = client.get_json(class_path)
        time.sleep(SLEEP_S)
        drug_info = (class_resp.get("rxclassDrugInfoList") or {}).get("rxclassDrugInfo") or []
        seen_class_ids = set()
        for item in drug_info:
            min_concept = item.get("rxclassMinConceptItem") or {}
            class_id = min_concept.get("classId")
            class_name = min_concept.get("className")
            if class_id and class_id not in seen_class_ids:
                seen_class_ids.add(class_id)
                atc_classes.append({"classId": class_id, "className": class_name})

    return {"rxcui": rxcui, "status": status, "atc_classes": atc_classes}


def atc_levels(class_id: str) -> dict:
    """Derive ATC levels 1-4 from a 5-character RxClass ATC classId by
    substring (level 5, the chemical substance, is not returned by RxClass)."""
    return {
        "atc1": class_id[0:1] if len(class_id) >= 1 else None,
        "atc2": class_id[0:3] if len(class_id) >= 3 else None,
        "atc3": class_id[0:4] if len(class_id) >= 4 else None,
        "atc4": class_id[0:5] if len(class_id) >= 5 else None,
    }


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1, sort_keys=True)
    os.replace(tmp_path, path)  # atomic on POSIX; avoids a truncated cache on interrupt


def build_crosswalk(cache_path: str, limit: int | None = None) -> dict:
    medrecon = load_medrecon()
    ndc_counts = Counter(medrecon["ndc"].dropna().astype(str))
    distinct_ndcs = sorted(ndc_counts.keys())
    if limit is not None:
        distinct_ndcs = distinct_ndcs[:limit]

    cache = load_cache(cache_path)
    to_resolve = [n for n in distinct_ndcs if n not in cache or cache[n].get("status") == "ERROR"]

    print(f"medrecon rows: {len(medrecon)}", flush=True)
    print(f"distinct NDCs: {len(distinct_ndcs)}  (cached already: {len(distinct_ndcs) - len(to_resolve)})", flush=True)
    print(f"to resolve this run: {len(to_resolve)}", flush=True)

    client = RxNavClient()
    resolved_since_flush = 0
    t_start = time.time()
    try:
        for i, ndc in enumerate(to_resolve):
            try:
                result = resolve_ndc(client, ndc)
            except RuntimeError as exc:
                result = {"rxcui": "", "status": "ERROR", "atc_classes": [], "error": str(exc)}
            cache[ndc] = result
            resolved_since_flush += 1
            if resolved_since_flush >= FLUSH_EVERY:
                save_cache(cache, cache_path)
                resolved_since_flush = 0
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                remaining_s = (len(to_resolve) - (i + 1)) / rate if rate > 0 else float("nan")
                print(
                    f"  [{i + 1}/{len(to_resolve)}] checkpointed ({len(cache)} total entries); "
                    f"{rate:.2f} NDC/s, ~{remaining_s / 60:.1f} min remaining",
                    flush=True,
                )
    finally:
        client.close()

    save_cache(cache, cache_path)
    return cache


def compute_coverage(cache: dict, medrecon_ndc_counts: Counter) -> dict:
    total_rows = sum(medrecon_ndc_counts.values())
    real_rows = total_rows - medrecon_ndc_counts.get(MIMIC_NULL_NDC, 0)

    mapped_ndcs = 0
    mapped_rows = 0
    etc_fallback_rows = 0
    atc1_row_counts: Counter = Counter()

    for ndc, count in medrecon_ndc_counts.items():
        if ndc == MIMIC_NULL_NDC:
            continue
        entry = cache.get(ndc, {})
        classes = entry.get("atc_classes") or []
        if classes:
            mapped_ndcs += 1
            mapped_rows += count
            for c in classes:
                lvl = atc_levels(c["classId"])
                if lvl["atc1"]:
                    atc1_row_counts[lvl["atc1"]] += count
        else:
            etc_fallback_rows += count

    distinct_real_ndcs = sum(1 for n in medrecon_ndc_counts if n != MIMIC_NULL_NDC)

    return {
        "total_medrecon_rows": total_rows,
        "sentinel_no_ndc_rows": medrecon_ndc_counts.get(MIMIC_NULL_NDC, 0),
        "real_ndc_rows": real_rows,
        "distinct_real_ndcs": distinct_real_ndcs,
        "distinct_ndcs_mapped_to_atc": mapped_ndcs,
        "distinct_ndc_coverage_rate": round(mapped_ndcs / distinct_real_ndcs, 4) if distinct_real_ndcs else 0.0,
        "rows_mapped_to_atc": mapped_rows,
        "row_weighted_coverage_rate": round(mapped_rows / real_rows, 4) if real_rows else 0.0,
        "rows_falling_back_to_etc": etc_fallback_rows,
        "row_weighted_etc_fallback_rate": round(etc_fallback_rows / real_rows, 4) if real_rows else 0.0,
        "atc_level1_row_distribution": dict(atc1_row_counts.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only resolve the first N distinct NDCs (for a quick dry run)")
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--coverage-path", default=DEFAULT_COVERAGE_PATH)
    args = parser.parse_args()

    cache = build_crosswalk(args.cache_path, limit=args.limit)

    medrecon = load_medrecon()
    ndc_counts = Counter(medrecon["ndc"].dropna().astype(str))
    coverage = compute_coverage(cache, ndc_counts)

    os.makedirs(os.path.dirname(args.coverage_path), exist_ok=True)
    with open(args.coverage_path, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2)

    print(json.dumps(coverage, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
