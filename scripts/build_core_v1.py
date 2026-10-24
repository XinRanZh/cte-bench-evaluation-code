"""Build the CTE-Bench-Core-v1 JSONL and metadata report.

This script is intentionally deterministic: it filters the existing
counterfactual item files by the shared CORE_V1_CELLS list and then audits
the reference sliding-window runs on the same release split.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from core_v1 import (
    CORE_V1_CELLS,
    catalog_cell,
    is_core_catalog_item,
    normalize_line_number_convention,
)
from reachability import check_suffix_causal_effect
import run_provenance as rp  # noqa: E402
import grader  # noqa: E402

ALLOW_MIXED = False          # set by --allow-mixed
PROVENANCE_WARNINGS: list[str] = []


COMPLETE_SLIDING_RUNS = {
    "ds_lite_sliding": "ds_lite_sliding_labeled.json",
    "kimi_k25_sliding": "kimi_k25_sliding_labeled.json",
    "sonnet46_sliding": "sonnet46_sliding.json",
}
# Expected configuration of each reference run (legacy files: the fields they declare).
EXPECTED_RUNS = {
    "ds_lite_sliding_labeled.json": {"model": ("DS-V4-Lite/sliding", "deepseek/deepseek-chat")},
    "kimi_k25_sliding_labeled.json": {"model": ("Kimi-K2.5/sliding", "bedrock/moonshotai.kimi-k2.5")},
    "sonnet46_sliding.json": {"model": ("bedrock/us.anthropic.claude-sonnet-4-6",
                                        "bedrock/global.anthropic.claude-sonnet-4-6")},
}
EXPECTED_PROTOCOL = {"strategy": "sliding", "window": 20, "suffix_budget": 40}
CORE_CATALOG_SHA = None   # sha256 of the Core-v1 JSONL written by this run (set in main)
POOL_ROWS: list[dict] = []  # the candidate-pool rows the reference runs answer (set in build_audit)


# The candidate pool: one file per service, written by gen_counterfactuals.py
# (not part of the dataset release). Other catalogs in the directory
# (core_v1*.jsonl, no_effect_*.jsonl, ...) are not part of the pool.
POOL_FILES = ("auth.jsonl", "bank.jsonl", "cart.jsonl", "filesystem.jsonl", "ratelimiter.jsonl",
              "reservation.jsonl")


def load_catalog(cf_dir: Path) -> list[dict]:
    rows = []
    for path in (cf_dir / name for name in POOL_FILES):
        for line_no, line in enumerate(path.open(), start=1):
            row = json.loads(line)
            row["_source_file"] = path.name
            row["_source_line"] = line_no
            rows.append(row)
    return rows


def write_core_jsonl(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = Counter()
    with out_path.open("w") as f:
        for row in rows:
            clean = {k: v for k, v in row.items() if not k.startswith("_")}
            clean = normalize_line_number_convention(clean)
            item_id = clean.get("item_id")
            if item_id:
                seen_ids[item_id] += 1
                if seen_ids[item_id] > 1:
                    clean["item_id"] = f"{item_id}__{seen_ids[item_id]}"
            f.write(json.dumps(clean, separators=(",", ":")) + "\n")


def load_result_items(results_dir: Path, filename: str) -> list[dict]:
    data = json.loads((results_dir / filename).read_text())
    desc = rp.describe_run(data, filename)
    rp.expect_config(desc, {**EXPECTED_PROTOCOL, **EXPECTED_RUNS.get(filename, {})}, "build_core_v1 reference run")
    prov = rp.check([desc], allow_mixed=ALLOW_MIXED, tool="build_core_v1",
                    extra_problems=rp.catalog_problems(desc, CORE_CATALOG_SHA, "the Core-v1 catalog"))
    if prov["mixed"]:
        PROVENANCE_WARNINGS.append(rp.warning_line(prov))
        print(rp.warning_line(prov))
    if POOL_ROWS:
        # every row must answer exactly one candidate-pool row (id, calls, targets)
        import score_paired as sp
        pool_ids = {r.get("item_id") for r in POOL_ROWS}
        view, orig = [], {}
        for r in data.get("per_item", []):
            if isinstance(r, dict):
                v = {**r, "item_id": pool_item_id(r.get("item_id"), pool_ids)}
                orig[id(v)] = r
                r = v
            view.append(r)
        try:
            bound = rp.bind_to_catalog(desc, {**data, "per_item": view}, POOL_ROWS, None,
                                       catalog_name="the candidate pool", require_complete=False)
        except sp.InvalidRunError as e:
            raise SystemExit(f"error: build_core_v1: {filename} does not match the candidate pool ({e})")
        return [grader.scoring_view(orig[id(r)]) for _, r in bound]
    # calls that returned a provider error without model output are not scored
    return [grader.scoring_view(r) for r in data.get("per_item", []) if r and "error" not in r]


def pool_item_id(item_id, pool_ids: set) -> str:
    """The pool id a result row answers. Some reference runs label the second
    occurrence of a repeated pool id as '<id>__2' (the rule write_core_jsonl
    applies); such a row answers a pool row with id '<id>' and is then joined
    on the content of its scored calls and targets."""
    base, sep, n = str(item_id).rpartition("__")
    if item_id not in pool_ids and sep and n.isdigit() and base in pool_ids:
        return base
    return item_id


def aggregate_value_match(items: list[dict]) -> float | None:
    suffix_total = sum(r.get("suffix_len", 0) for r in items)
    if not suffix_total:
        return None
    return sum(r.get("n_value_match", 0) for r in items) / suffix_total


def summarize_cell_metrics(results_dir: Path) -> dict:
    runs = {
        name: load_result_items(results_dir, filename)
        for name, filename in COMPLETE_SLIDING_RUNS.items()
    }
    n = len(next(iter(runs.values())))
    for name, items in runs.items():
        if len(items) != n:
            raise ValueError(f"run length mismatch for {name}: {len(items)} != {n}")

    cell_rows = defaultdict(lambda: defaultdict(list))
    for i in range(n):
        cells = {(items[i]["domain"], items[i]["intervention_kind"])
                 for items in runs.values()}
        if len(cells) != 1:
            raise ValueError(f"run alignment mismatch at row {i}: {cells}")
        cell = next(iter(cells))
        for name, items in runs.items():
            cell_rows[cell][name].append(items[i])

    out = {}
    for cell in sorted(cell_rows):
        per_model = {
            name: aggregate_value_match(items)
            for name, items in cell_rows[cell].items()
        }
        values = [v for v in per_model.values() if v is not None]
        out[f"{cell[0]}::{cell[1]}"] = {
            "n_items": len(next(iter(cell_rows[cell].values()))),
            "per_model_value_match": per_model,
            "mean_value_match": sum(values) / len(values),
            "model_spread": max(values) - min(values),
        }
    return out


def summarize_run(results_dir: Path, cells: set[tuple[str, str]]) -> dict:
    out = {}
    for name, filename in COMPLETE_SLIDING_RUNS.items():
        items = [
            r for r in load_result_items(results_dir, filename)
            if (r["domain"], r["intervention_kind"]) in cells
        ]
        out[name] = {
            "n_items": len(items),
            "suffix_total": sum(r["suffix_len"] for r in items),
            "value_match": aggregate_value_match(items),
            "infra_excluded_calls": sum(sum((r.get("n_infra_excluded") or {}).values()) for r in items),
        }
    return out


def build_audit(catalog_rows: list[dict], core_rows: list[dict],
                results_dir: Path) -> dict:
    POOL_ROWS[:] = [{k: v for k, v in r.items() if not k.startswith("_")} for r in catalog_rows]
    core_cells = set(CORE_V1_CELLS)
    core_counts = Counter(catalog_cell(r) for r in core_rows)
    visible_effect = Counter()
    for row in core_rows:
        cell = catalog_cell(row)
        if check_suffix_causal_effect(row).accepted:
            visible_effect[cell] += 1

    core_pool = summarize_run(results_dir, core_cells)
    core_values = [v["value_match"] for v in core_pool.values()]

    cell_metrics = summarize_cell_metrics(results_dir)
    cell_report = {
        f"{d}::{k}": {
            "n_items": core_counts[(d, k)],
            "visible_effect_items": visible_effect[(d, k)],
            **cell_metrics[f"{d}::{k}"],
        }
        for d, k in CORE_V1_CELLS
    }

    return {
        "core_v1_cells": [{"domain": d, "kind": k} for d, k in CORE_V1_CELLS],
        "core_v1": {
            "n_items": len(core_rows),
            "suffix_decisions_per_configuration": len(core_rows) * 40,
            "cell_counts": {f"{d}::{k}": n for (d, k), n in sorted(core_counts.items())},
            "complete_sliding_value_match": core_pool,
            "mean_value_match": sum(core_values) / len(core_values),
            "model_spread": max(core_values) - min(core_values),
            "cell_metrics": cell_report,
        },
        "audit_notes": [
            "Adversarial delayed-effect patches are outside Core-v1 and kept as a separate frontier probe.",
            "All Core-v1 rows pass the suffix-visible causal-effect filter.",
            "Core-v1 source strings are normalized so line 1 is the first non-empty source line.",
        ],
    }


def preflight(cf_dir: Path, results_dir: Path) -> None:
    """Stop before writing anything unless the candidate pool and every
    reference run are present."""
    missing = [n for n in POOL_FILES if not (cf_dir / n).is_file()]
    if missing:
        raise SystemExit(f"error: build_core_v1: {cf_dir} is not the candidate pool (missing "
                         f"{', '.join(missing)}). The pool is the six per-service files written by "
                         "scripts/gen_counterfactuals.py; it is not part of the dataset release. Nothing was written.")
    missing = [f for f in COMPLETE_SLIDING_RUNS.values() if not (results_dir / f).is_file()]
    if missing:
        raise SystemExit(f"error: build_core_v1: reference runs missing in {results_dir}: "
                         f"{', '.join(missing)}. Nothing was written.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", type=Path, default=Path("counterfactuals"),
                    help="directory with the candidate pool (the six per-service files)")
    ap.add_argument("--results-dir", type=Path, default=Path("results/full_grid_2026-04-29"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("build/core_v1.jsonl"),
                    help="output catalog (default build/core_v1.jsonl; the released catalog is never "
                         "touched unless given here, and only replaced after a successful build)")
    ap.add_argument("--audit-json", type=Path, default=Path("build/core_v1_audit.json"))
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    args = ap.parse_args()
    global ALLOW_MIXED, CORE_CATALOG_SHA
    ALLOW_MIXED = args.allow_mixed

    preflight(args.cf_dir, args.results_dir)
    catalog_rows = load_catalog(args.cf_dir)
    core_rows = [r for r in catalog_rows if is_core_catalog_item(r)]
    # Build into temporary files next to the outputs; replace the outputs only
    # after the whole build (catalog and audit) has succeeded.
    tmps = []
    try:
        for out in (args.out_jsonl, args.audit_json):
            out.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", dir=out.parent)
            os.close(fd)
            tmps.append(Path(tmp))
        tmp_core, tmp_audit = tmps
        write_core_jsonl(core_rows, tmp_core)
        CORE_CATALOG_SHA = rp.file_sha256(tmp_core)
        try:
            audit = build_audit(catalog_rows, core_rows, args.results_dir)
        except rp.MixedRunError as e:
            raise SystemExit(f"error: {e}")
        if PROVENANCE_WARNINGS:
            audit["mixed_run_provenance"] = PROVENANCE_WARNINGS
        tmp_audit.write_text(json.dumps(audit, indent=2) + "\n")
        os.replace(tmp_core, args.out_jsonl)
        os.replace(tmp_audit, args.audit_json)
    finally:
        for t in tmps:
            if t.exists():
                t.unlink()

    print(f"wrote {args.out_jsonl} ({len(core_rows)} rows)")
    print(f"wrote {args.audit_json}")


if __name__ == "__main__":
    main()
