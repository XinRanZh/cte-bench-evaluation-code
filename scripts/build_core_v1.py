"""Build the CTE-Bench-Core-v1 JSONL and metadata report.

This script is intentionally deterministic: it filters the existing
counterfactual item files by the shared CORE_V1_CELLS list and then audits
the reference sliding-window runs on the same release split.
"""
from __future__ import annotations

import argparse
import json
import sys
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


COMPLETE_SLIDING_RUNS = {
    "ds_lite_sliding": "ds_lite_sliding_labeled.json",
    "kimi_k25_sliding": "kimi_k25_sliding_labeled.json",
    "sonnet46_sliding": "sonnet46_sliding.json",
}


def load_catalog(cf_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(cf_dir.glob("*.jsonl")):
        if path.name == "core_v1.jsonl":
            continue
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
    return [r for r in data.get("per_item", []) if r and "error" not in r]


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
        }
    return out


def build_audit(catalog_rows: list[dict], core_rows: list[dict],
                results_dir: Path) -> dict:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", type=Path, default=Path("counterfactuals"))
    ap.add_argument("--results-dir", type=Path, default=Path("results/full_grid_2026-04-29"))
    ap.add_argument("--out-jsonl", type=Path, default=Path("counterfactuals/core_v1.jsonl"))
    ap.add_argument("--audit-json", type=Path, default=Path("dataset_metadata/core_v1_audit.json"))
    args = ap.parse_args()

    catalog_rows = load_catalog(args.cf_dir)
    core_rows = [r for r in catalog_rows if is_core_catalog_item(r)]
    write_core_jsonl(core_rows, args.out_jsonl)
    audit = build_audit(catalog_rows, core_rows, args.results_dir)
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"wrote {args.out_jsonl} ({len(core_rows)} rows)")
    print(f"wrote {args.audit_json}")


if __name__ == "__main__":
    main()
