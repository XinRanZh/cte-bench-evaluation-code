"""Shared definition of the CTE-Bench-Core-v1 slice.

Core-v1 is the paper-facing benchmark split. It retains the domain and
intervention cells used in the NeurIPS E&D draft, with coverage across all
six services and four non-adversarial intervention labels. The adversarial
delayed-effect patches remain a separate frontier probe.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
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

# ---------------------------------------------------------------------------
# No-effect v1: the matched benign-patch companion split, 1:1 paired with
# Core-v1 (`counterfactuals/no_effect_v1.jsonl`).
#
# Built by `scripts/build_no_effect_v1.py`; scored together with Core-v1 by
# `scripts/score_paired.py` (paired reporting is required, see README).
# Rows are matched to Core-v1 on domain, trajectory and prefix_len, but NOT
# on intervention kind: a benign patch of the partner's own kind does not
# always exist (e.g. 0/30 clock_jump rows admit a benign clock value), so
# 93/255 rows (~36%) carry a near-kind patch.
# Address the split by (domain, kind) cell only for diagnostics; use
# `paired_item_id` for anything that must line up with Core-v1 row-for-row.
NO_EFFECT_V1_CELLS: tuple[tuple[str, str], ...] = (
    ("auth", "threshold_change"),
    ("bank", "branch_inversion"),
    ("bank", "threshold_change"),
    ("cart", "branch_inversion"),
    ("cart", "threshold_change"),
    ("filesystem", "branch_inversion"),
    ("filesystem", "threshold_change"),
    ("ratelimiter", "branch_inversion"),
    ("ratelimiter", "state_injection"),
    ("ratelimiter", "threshold_change"),
    ("reservation", "branch_inversion"),
    ("reservation", "threshold_change"),
)

NO_EFFECT_V1_CELL_SET = set(NO_EFFECT_V1_CELLS)

BENIGN_TIERS = ("same_kind", "near_kind", "equivalent")
CONTINGENCY_CLASSES = ("contingent", "global_no_op", "equivalent")

CORE_V1_JSONL = "counterfactuals/core_v1.jsonl"
NO_EFFECT_V1_JSONL = "counterfactuals/no_effect_v1.jsonl"
NO_EFFECT_V0_JSONL = "counterfactuals/no_effect_v0.jsonl"

SPLITS = {
    "core_v1": CORE_V1_JSONL,
    "no_effect_v1": NO_EFFECT_V1_JSONL,
    "no_effect_v0": NO_EFFECT_V0_JSONL,
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_absolute():
        path = _repo_root() / path
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_split(name: str) -> list[dict]:
    """Load a named release split ('core_v1' | 'no_effect_v1' | 'no_effect_v0')."""
    if name not in SPLITS:
        raise KeyError(f"unknown split {name!r}; known: {sorted(SPLITS)}")
    return load_jsonl(SPLITS[name])


def load_core_v1() -> list[dict]:
    return load_split("core_v1")


def load_no_effect_v1() -> list[dict]:
    return load_split("no_effect_v1")


def is_no_effect_v1_catalog_item(item: dict) -> bool:
    return catalog_cell(item) in NO_EFFECT_V1_CELL_SET


def is_no_effect_v1_result_item(item: dict) -> bool:
    return result_cell(item) in NO_EFFECT_V1_CELL_SET


def filter_no_effect_v1_results(items: Iterable[dict]) -> list[dict]:
    return [r for r in items
            if r and "error" not in r and is_no_effect_v1_result_item(r)]


def pair_index(no_effect_rows: Iterable[dict]) -> dict[str, dict]:
    """Core-v1 item_id -> its paired no-effect v1 row."""
    return {r["paired_item_id"]: r for r in no_effect_rows
            if r.get("paired_item_id")}


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


CORE_V1_CATALOG = Path(__file__).resolve().parent.parent / "counterfactuals" / "core_v1.jsonl"
CORE_V1_SHA256 = "8be8b32fbb4ad8123861353045356dcea00a257b970e45eb7d0ac322d91ded72"  # the released catalog


def _run_provenance():
    import importlib.util
    import sys
    if "run_provenance" in sys.modules:
        return sys.modules["run_provenance"]
    spec = importlib.util.spec_from_file_location("run_provenance", Path(__file__).with_name("run_provenance.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_provenance"] = mod
    spec.loader.exec_module(mod)
    return mod


def filter_core_results(items: Iterable[dict], allow_mixed: bool = False,
                        catalog: Path | None = None) -> list[dict]:
    """Core-v1 rows of an evaluator output. Rows stamped with different run
    configurations (`run_config`), or stamped mixed with unstamped rows, are
    rejected unless allow_mixed (see scripts/run_provenance.py). Every returned
    row must answer one row of the Core-v1 catalog (default: the released
    counterfactuals/core_v1.jsonl) with the same id, calls and targets;
    otherwise score_paired.InvalidRunError is raised."""
    items = list(items)
    rp = _run_provenance()
    import sys
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import score_paired as sp
    desc = rp.describe_run({"per_item": items}, "per_item rows")
    rp.check([desc], allow_mixed=allow_mixed, tool="core_v1.filter_core_results")
    rows = [r for r in items if r and "error" not in r and is_core_result_item(r)]
    catalog = Path(catalog) if catalog is not None else CORE_V1_CATALOG
    rp.bind_to_catalog(desc, {"per_item": rows}, sp.load_jsonl(catalog), None,
                       catalog_name=catalog.name, require_complete=False)
    return rows


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
