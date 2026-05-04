"""Graders for CTE-Bench evaluation.

Pure functions, no network, no LLM. Consumed by:
  - scripts/eval_llm_counterfactual.py  (per-item scoring during eval)
  - scripts/leaderboard.py              (re-scoring for aggregation / sanity)

Four graders, each matching a headline metric:

  1. `grade_value_match(predicted_response, ground_truth_response)`
     Per-step exact-match of decoded LLM JSON vs. the counterfactual ground
     truth.

  2. `grade_status_match(predicted, gt)`
     Error-vs-success polarity match. A coarser metric that's still useful
     when the LLM's literal payload drifts but the branch is correct.

  3. `grade_explanation_localization(predicted_line, explanation_truth,
                                      tolerance=1)`
     Exact-line / adjacent-line grading of the LLM's claim about which
     source line was intervened on. Tolerance of 0 = exact; 1 = one-line
     neighbourhood. We also report the absolute line distance as a
     histogram-friendly number. Core-v1 uses exact or one-line diagnostics;
     wider tolerances are too permissive for headline reporting.

  4. `grade_state_probe(predicted_state, ground_truth_state, path_set=None)`
     Subset-equality on selected state keys. For adversarial items where the
     response-level value-match is zero but the underlying hidden state
     diverges (e.g. bank_withdrawal_sign), this is the only signal.

All graders return small structured dicts. They never raise on malformed
predictions — parse failures become `{"valid": False, "reason": ...}`.
"""
from __future__ import annotations
import json
import re
from typing import Any, Iterable, Optional


# ---------------------------------------------------------------------------
# Loose JSON parsing from LLM output (mirrors eval_llm_baseline.parse_json_loose
# with two additions: tolerate trailing commas, strip BOM).

def parse_json_loose(s: str) -> Any:
    if not isinstance(s, str):
        return None
    s = s.strip().lstrip("﻿")
    if not s:
        return None
    # strip markdown fence
    if s.startswith("```"):
        # remove opening fence + optional lang tag
        m = re.match(r"```[a-zA-Z]*\s*\n?", s)
        if m:
            s = s[m.end():]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    # grab first balanced {...}
    depth = 0; start = None; end = None
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                end = i + 1
                break
    if start is None or end is None:
        return None
    blob = s[start:end]
    # tolerate trailing commas
    blob = re.sub(r",(\s*[}\]])", r"\1", blob)
    try:
        return json.loads(blob)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. value-match grader

def grade_value_match(predicted: Any, ground_truth: Any) -> dict:
    if predicted is None:
        return {"match": False, "reason": "PARSE_FAIL",
                "well_formed": False}
    if not isinstance(predicted, dict):
        return {"match": False, "reason": "NOT_OBJECT",
                "well_formed": False}
    ok = predicted == ground_truth
    return {"match": ok,
            "reason": "OK" if ok else "MISMATCH",
            "well_formed": True,
            "predicted_keys": sorted(predicted.keys()),
            "ground_truth_keys": sorted(ground_truth.keys())
                if isinstance(ground_truth, dict) else None}


# ---------------------------------------------------------------------------
# 2. status-match grader

def _is_error(resp: Any) -> bool:
    return isinstance(resp, dict) and "error" in resp


def grade_status_match(predicted: Any, ground_truth: Any) -> dict:
    if predicted is None:
        return {"match": False, "reason": "PARSE_FAIL"}
    pred_err = _is_error(predicted)
    gt_err   = _is_error(ground_truth)
    return {"match": pred_err == gt_err,
            "predicted_error": pred_err,
            "ground_truth_error": gt_err,
            "predicted_code": (predicted.get("error", {}).get("code")
                               if pred_err and isinstance(predicted, dict) else None),
            "ground_truth_code": (ground_truth.get("error", {}).get("code")
                                  if gt_err and isinstance(ground_truth, dict) else None)}


# ---------------------------------------------------------------------------
# 3. explanation-localization grader

_LINE_HINT_PATTERNS = [
    re.compile(r"\bline[s]?\s*[:#=]?\s*(\d+)", re.I),
    re.compile(r"line\s+number[:=]?\s*(\d+)", re.I),
    re.compile(r"^\s*(\d+)\s*[:.)]", re.M),
]


def extract_line_number(response_text: str) -> Optional[int]:
    """Parse a line number out of free-form LLM text, or from JSON with a
    `line_number` / `line` / `lineno` key. Returns None if none found.
    """
    if not response_text:
        return None
    parsed = parse_json_loose(response_text)
    if isinstance(parsed, dict):
        for k in ("line_number", "line", "lineno", "line_no"):
            if k in parsed and isinstance(parsed[k], (int, str)):
                try:
                    return int(parsed[k])
                except (ValueError, TypeError):
                    continue
    for pat in _LINE_HINT_PATTERNS:
        m = pat.search(response_text)
        if m:
            try:
                return int(m.group(1))
            except (ValueError, TypeError):
                continue
    return None


def grade_explanation_localization(
        predicted: int | str | None,
        explanation_truth: dict,
        tolerance: int = 1) -> dict:
    """Grade a predicted line number against the ground truth.

    Args:
      predicted: either an int, a string we try to extract a line number from,
        or None. For free-form LLM text, call `extract_line_number` first
        (we also call it defensively on string inputs here).
      explanation_truth: the item's `explanation_truth` dict; must carry
        `line_number` for code/adversarial interventions. State-injection
        items carry `path` instead — we return `applicable: False` in that
        case.
      tolerance: max |predicted - truth| to count as "adjacent match".
        Use tolerance=1 for the released diagnostic; tolerance=2 is useful
        only for manual audit because several edits are only a few lines apart.
    """
    if not isinstance(explanation_truth, dict):
        return {"applicable": False, "reason": "NO_TRUTH"}
    truth_line = explanation_truth.get("line_number")
    if truth_line is None:
        return {"applicable": False,
                "reason": "STATE_LEVEL_INTERVENTION",
                "path": explanation_truth.get("path")}

    pred_int: Optional[int]
    if isinstance(predicted, int):
        pred_int = predicted
    elif isinstance(predicted, str):
        pred_int = extract_line_number(predicted)
    else:
        pred_int = None

    if pred_int is None:
        return {"applicable": True, "extracted": None,
                "exact_match": False, "adjacent_match": False,
                "line_distance": None, "reason": "NO_LINE_EXTRACTED"}

    distance = abs(pred_int - int(truth_line))
    return {"applicable": True, "extracted": pred_int,
            "truth_line": int(truth_line),
            "line_distance": distance,
            "exact_match": distance == 0,
            "adjacent_match": distance <= tolerance,
            "reason": "OK"}


# ---------------------------------------------------------------------------
# 4. state-probe grader

def _deep_eq(a: Any, b: Any, float_tol: float = 1e-9) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return abs(float(a) - float(b)) <= float_tol
        except (TypeError, ValueError):
            return False
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        return all(_deep_eq(a[k], b[k], float_tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_deep_eq(x, y, float_tol) for x, y in zip(a, b))
    return a == b


def _extract_paths(obj: Any, prefix: Optional[list] = None,
                   out: Optional[list] = None) -> list[tuple]:
    if prefix is None:
        prefix = []
    if out is None:
        out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            _extract_paths(v, prefix + [k], out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _extract_paths(v, prefix + [i], out)
    else:
        out.append(tuple(prefix))
    return out


def _walk(obj: Any, path: Iterable) -> Any:
    cur = obj
    for step in path:
        if isinstance(cur, dict):
            cur = cur.get(step)
        elif isinstance(cur, list):
            try:
                cur = cur[int(step)] if 0 <= int(step) < len(cur) else None
            except (TypeError, ValueError):
                return None
        else:
            return None
    return cur


def grade_state_probe(predicted_state: Any, ground_truth_state: Any,
                      path_set: Optional[Iterable] = None) -> dict:
    """Per-path agreement between predicted and ground-truth state dicts.

    If `path_set` is None, we enumerate all scalar paths appearing in the
    ground truth and grade each. Returns per-path match + aggregate summary.
    """
    if not isinstance(predicted_state, dict) and path_set is None:
        return {"valid": False, "reason": "PREDICTION_NOT_OBJECT"}
    if not isinstance(ground_truth_state, dict):
        return {"valid": False, "reason": "GROUND_TRUTH_NOT_OBJECT"}

    if path_set is None:
        paths = _extract_paths(ground_truth_state)
    else:
        paths = [tuple(p) for p in path_set]

    per_path = []
    n_match = 0
    for p in paths:
        gt_v = _walk(ground_truth_state, p)
        pr_v = _walk(predicted_state, p)
        ok = _deep_eq(pr_v, gt_v)
        per_path.append({"path": list(p), "match": ok,
                         "predicted": pr_v, "ground_truth": gt_v})
        if ok:
            n_match += 1

    return {"valid": True,
            "n_paths": len(paths),
            "n_match": n_match,
            "fraction_match": (n_match / len(paths)) if paths else 0.0,
            "per_path": per_path}


def _walk_probe_path(state: Any, path: list):
    """Walk a v2 probe path. Supports two special opcodes as path[0]:
      '__len__'   — len(state_after_walking_rest)
      '__sum_qty__' — sum of 'qty' fields in a list of dicts
    Otherwise acts like _walk.
    """
    if path and path[0] == "__len__":
        rest = path[1:]
        v = _walk(state, rest) if rest else state
        if v is None:
            return None
        try:
            return len(v)
        except TypeError:
            return None
    if path and path[0] == "__sum_qty__":
        rest = path[1:]
        v = _walk(state, rest)
        if not isinstance(v, list):
            return None
        try:
            return sum(int(x.get("qty", 0)) for x in v if isinstance(x, dict))
        except (TypeError, ValueError):
            return None
    return _walk(state, path)


def _expand_probe_paths(probe_spec: list, state: dict) -> list[list]:
    """Expand PROBE_PATHS_V2 entries against `state`, returning concrete paths."""
    out = []
    for entry in probe_spec:
        if isinstance(entry, tuple) and len(entry) == 3 and entry[0] == "dynamic":
            _, _label, fn = entry
            try:
                for p in fn(state or {}):
                    if p and list(p) not in [list(x) for x in out]:
                        out.append(list(p))
            except Exception:
                continue
        elif isinstance(entry, list):
            if entry not in out:
                out.append(list(entry))
    return out


def grade_state_probe_v2(predicted_state: Any,
                         ground_truth_cf: Any,
                         ground_truth_null: Any,
                         probe_spec: list) -> dict:
    """State-probe v2 with predeclared paths + null-baseline subtraction.

    Given a testbed-level `probe_spec` (PROBE_PATHS_V2), expand dynamic
    paths against the counterfactual ground-truth state, walk each path on
    predicted / counterfactual-oracle / null-oracle (same trace under the
    ORIGINAL program), and score each path as one of:

      - intervention_effective=False: cf value == null value.  These paths
        are dropped from scoring: they carry no signal about whether the
        model predicted the counterfactual effect; scoring them rewards
        echoing the prefix.
      - intervention_effective=True AND predicted == cf: credit.
      - intervention_effective=True AND predicted != cf: no credit.

    The aggregate is computed only over effective paths (`n_effective`).
    `n_match` is the count of effective paths where predicted == cf.
    """
    if not isinstance(ground_truth_cf, dict) or not isinstance(ground_truth_null, dict):
        return {"valid": False, "reason": "GROUND_TRUTH_MISSING"}
    if not isinstance(predicted_state, dict):
        return {"valid": False, "reason": "PREDICTION_NOT_OBJECT"}

    paths = _expand_probe_paths(probe_spec, ground_truth_cf)

    per_path = []
    n_effective = 0
    n_match = 0
    n_total_declared = len(paths)
    for p in paths:
        gt_cf = _walk_probe_path(ground_truth_cf, p)
        gt_null = _walk_probe_path(ground_truth_null, p)
        pr = _walk_probe_path(predicted_state, p)
        effective = not _deep_eq(gt_cf, gt_null)
        match = _deep_eq(pr, gt_cf) if effective else None
        if effective:
            n_effective += 1
            if match:
                n_match += 1
        per_path.append({"path": list(p),
                         "predicted": pr,
                         "gt_cf": gt_cf,
                         "gt_null": gt_null,
                         "effective": effective,
                         "match": match})

    return {
        "valid": True,
        "n_declared_paths": n_total_declared,
        "n_effective": n_effective,
        "n_match": n_match,
        "fraction_match": (n_match / n_effective) if n_effective else None,
        "per_path": per_path,
    }
