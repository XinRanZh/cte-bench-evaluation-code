"""Shared definition of the CTE-Bench-Core-v1 slice.

Core-v1 is the paper-facing benchmark split. It retains the domain and
intervention cells used in the NeurIPS E&D draft, with coverage across all
six services and four non-adversarial intervention labels. The adversarial
delayed-effect patches remain a separate frontier probe.
"""
from __future__ import annotations

import copy
import re
from typing import Iterable


# Ordered for stable JSONL generation and paper tables.
CORE_V1_CELLS: tuple[tuple[str, str], ...] = (
    ("auth", "threshold_change"),
    ("bank", "branch_inversion"),
    ("cart", "branch_inversion"),
    ("filesystem", "branch_inversion"),
    ("ratelimiter", "branch_inversion"),
    ("ratelimiter", "clock_jump"),
    ("ratelimiter", "state_injection"),
    ("ratelimiter", "threshold_change"),
    ("reservation", "branch_inversion"),
)

CORE_V1_CELL_SET = set(CORE_V1_CELLS)


def catalog_cell(item: dict) -> tuple[str, str]:
    """Return (domain, intervention_kind) for a counterfactual catalog row."""
    intervention = item.get("intervention") or {}
    return item.get("domain"), intervention.get("kind")


def result_cell(item: dict) -> tuple[str, str]:
    """Return (domain, intervention_kind) for an eval result row."""
    return item.get("domain"), item.get("intervention_kind")


def is_core_catalog_item(item: dict) -> bool:
    return catalog_cell(item) in CORE_V1_CELL_SET


def is_core_result_item(item: dict) -> bool:
    return result_cell(item) in CORE_V1_CELL_SET


def filter_core_results(items: Iterable[dict]) -> list[dict]:
    return [r for r in items if r and "error" not in r and is_core_result_item(r)]


def _leading_newline_count(text: object) -> int:
    if not isinstance(text, str):
        return 0
    return len(text) - len(text.lstrip("\n"))


def normalize_source_text(source: object) -> object:
    """Canonicalize source shown to models for line-number tasks.

    The original testbed SOURCE_CODE strings were triple-quoted with a leading
    newline, which made the visual first code line appear as line 2. Core-v1
    uses the cleaner convention: line 1 is the first non-empty source line.
    """
    if not isinstance(source, str):
        return source
    return source.lstrip("\n")


def normalize_line_number_convention(item: dict) -> dict:
    """Return a copy with source strings and line-number labels normalized.

    This preserves item IDs and oracle traces. It only removes leading blank
    lines from displayed source strings and shifts code-intervention line
    labels by the same offset. State-intervention rows have no source line
    labels and are unaffected except for source display cleanup when present.
    """
    out = copy.deepcopy(item)
    source_offsets = []
    for field in ("source_code", "patched_source"):
        offset = _leading_newline_count(out.get(field))
        if offset:
            out[field] = out[field][offset:]
            source_offsets.append(offset)

    offset = max(source_offsets) if source_offsets else 0
    if not offset:
        return out

    for field in ("intervention", "explanation_truth"):
        record = out.get(field)
        if not isinstance(record, dict) or "line_number" not in record:
            continue
        old_line = int(record["line_number"])
        new_line = max(1, old_line - offset)
        record["line_number"] = new_line
        if field == "intervention" and isinstance(record.get("description"), str):
            record["description"] = re.sub(
                rf"\bline\s+{old_line}\b",
                f"line {new_line}",
                record["description"],
            )
    return out
