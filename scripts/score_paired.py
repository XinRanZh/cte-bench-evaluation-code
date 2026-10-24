"""Paired scorer: Core-v1 effect-step VM together with no-effect-v1 VM.

CTE-Bench-Core-v1 only contains scenarios in which the intervention changes at
least one future response, so effect-step VM rewards predicting changes that
do happen but never penalizes inventing changes that do not happen. The
no-effect-v1 companion split (`counterfactuals/no_effect_v1.jsonl`) pairs every
Core-v1 scenario with one scenario on the same service, the same observed
history and the same 40 future calls, carrying a benign patch whose future
responses are identical to the unpatched service. The two coordinates are

    Eff VM  exact value match on Core-v1 effect steps (2,476 steps, 247 scenarios)
    NE VM   exact value match on all 10,200 scored no-effect-v1 steps (255 scenarios)

PAIRED REPORTING IS REQUIRED. Report both coordinates, each with its interval.
Do not collapse them into min(), a mean, or a difference used as a ranking:
a model that ignores every intervention scores NE VM = 100 and Eff VM = 0, and
min(Eff VM, NE VM) equals Eff VM for both reference models, so a single
number either hides the no-effect axis or rewards ignoring interventions.

Joins (both validated against the released catalogs, never trusted blindly):
  * no-effect side: by `item_id` (no-effect rows share null-world traces, so a
    schedule join would merge rows: 255 rows have only 240 distinct
    40-step schedules);
  * Core-v1 side: by the 40-step (op, args, ground truth) schedule, which is
    unique over the 255 Core-v1 rows. This works for the released 255-item
    reference outputs and for broader development runs that stored variants
    under a base `item_id`.
Every step is re-graded against the catalog oracle with `grader.grade_value_match`;
disagreements with the run file's stored `value_match` flag are counted and
reported. The stored flags are never used for scoring: a step without a
`predicted` field is rejected. Each scored suffix index must occur exactly once
and lie inside the scored window, and every step's call (op, args) must equal
the catalog's call at that index; violations raise `InvalidRunError`.

Bootstrap: scenario resampling, ratio of sums, percentile interval, default
B=2000 and seed=17, items in catalog order (so the interval does not depend on
the order of rows in a run file).

Usage:
    python scripts/score_paired.py \
        --core-run reference_results/raw/deepseek_v4_flash_tf1step_sliding_k20.json \
        --no-effect-run reference_results/no_effect_v1_raw/deepseek_v4_flash_no_effect_v1_tf1step_sliding_k20.json \
        --label "DeepSeek V4-Flash" --json scratch/paired.json

    # several models at once
    python scripts/score_paired.py \
        --pair "DeepSeek V4-Flash" CORE_RUN_DS NE_RUN_DS \
        --pair "Kimi K2.5" CORE_RUN_KIMI NE_RUN_KIMI --json scratch/paired.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from grader import grade_status_match, grade_value_match, is_infrastructure_failure  # noqa: E402
import run_provenance as rp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORE = ROOT / "counterfactuals" / "core_v1.jsonl"
DEFAULT_NO_EFFECT = ROOT / "counterfactuals" / "no_effect_v1.jsonl"
SUFFIX_BUDGET = 40

REPORTING_RULE = (
    "Report the pair (Core-v1 effect-step VM, no-effect-v1 VM), each with its "
    "scenario-bootstrap interval, under one named protocol. Do not collapse the "
    "pair into min(), a mean, or a difference. NE VM alone can be inflated by "
    "ignoring interventions (a null-world predictor scores NE VM 100, Eff VM 0)."
)


# ---------------------------------------------------------------------------
# Loading and validation

def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_run(path: Path) -> dict:
    with Path(path).open() as f:
        data = json.load(f)
    if not isinstance(data, dict) or "per_item" not in data:
        raise ValueError(f"{path}: not an eval_llm_counterfactual result file")
    return data


def schedule_key_from_catalog(row: dict, budget: int) -> tuple:
    return tuple((s["op"], canon(s["args"]), canon(s["response"]))
                 for s in row["trace_counterfactual"][:budget])


class InvalidRunError(ValueError):
    """A run file that cannot be scored without trusting unverified fields."""


def _valid_index(j) -> bool:
    return isinstance(j, int) and not isinstance(j, bool) and j >= 0


def scored_steps(run_row: dict, budget: int) -> list[dict]:
    """Steps inside the scored window, sorted by suffix index (lenient; used
    only to build join keys over arbitrary run rows)."""
    steps = [s for s in (run_row.get("phase_a") or [])
             if _valid_index(s.get("suffix_idx")) and s["suffix_idx"] < budget]
    return sorted(steps, key=lambda s: s["suffix_idx"])


def checked_steps(run_row: dict, catalog_row: dict, budget: int) -> list[dict]:
    """Strictly validated scored steps of a run row joined to `catalog_row`.

    Raises InvalidRunError if any step has a missing, negative or non-integer
    suffix index, an index outside the stored trace, a duplicate index inside
    the scored window, a call (op, args) that differs from the catalog's call at
    that index, or no `predicted` field (flag-only results are not accepted).
    """
    iid = run_row.get("item_id")
    trace = catalog_row["trace_counterfactual"]
    seen: set[int] = set()
    out = []
    for s in run_row.get("phase_a") or []:
        j = s.get("suffix_idx")
        if not _valid_index(j):
            raise InvalidRunError(f"{iid}: invalid suffix_idx {j!r}")
        if j >= len(trace):
            raise InvalidRunError(f"{iid}: suffix_idx {j} outside the stored trace ({len(trace)} calls)")
        if j >= budget:
            continue
        if j in seen:
            raise InvalidRunError(f"{iid}: duplicate suffix_idx {j}")
        seen.add(j)
        if s.get("op") != trace[j]["op"] or canon(s.get("args")) != canon(trace[j]["args"]):
            raise InvalidRunError(f"{iid}: call at suffix_idx {j} differs from the catalog")
        if "predicted" not in s:
            raise InvalidRunError(f"{iid}: step {j} has no 'predicted' field; "
                                  "flag-only result files cannot be re-graded")
        out.append(s)
    return sorted(out, key=lambda s: s["suffix_idx"])


def schedule_key_from_run(run_row: dict, budget: int) -> tuple:
    return tuple((s["op"], canon(s["args"]), canon(s["ground_truth"]))
                 for s in scored_steps(run_row, budget))


def validate_pairing(core_rows: list[dict], ne_rows: list[dict], budget: int) -> dict:
    """Structural checks that make the two coordinates comparable."""
    core_by_id = {r["item_id"]: r for r in core_rows}
    problems = []
    if len(core_by_id) != len(core_rows):
        problems.append("duplicate item_id in Core-v1 catalog")
    ne_ids = [r["item_id"] for r in ne_rows]
    if len(set(ne_ids)) != len(ne_ids):
        problems.append("duplicate item_id in no-effect catalog")
    paired = [r.get("paired_item_id") for r in ne_rows]
    if len(set(paired)) != len(paired):
        problems.append("paired_item_id is not 1:1")
    missing = set(core_by_id) - set(paired)
    if missing:
        problems.append(f"{len(missing)} Core-v1 rows have no companion")
    n_pos = n_pos_diff = n_prefix_same = n_sched_same = 0
    for r in ne_rows:
        c = core_by_id.get(r.get("paired_item_id"))
        if c is None:
            problems.append(f"{r['item_id']}: unknown paired_item_id")
            continue
        if (r["prefix_len"] == c["prefix_len"]
                and canon(r["trace_prefix"]) == canon(c["trace_prefix"])):
            n_prefix_same += 1
        sched_ne = [(s["op"], canon(s["args"])) for s in r["trace_counterfactual"][:budget]]
        sched_c = [(s["op"], canon(s["args"])) for s in c["trace_counterfactual"][:budget]]
        if sched_ne == sched_c:
            n_sched_same += 1
        for a, b in zip(r["trace_original"][:budget], r["trace_counterfactual"][:budget]):
            n_pos += 1
            n_pos_diff += canon(a["response"]) != canon(b["response"])
    if n_pos_diff:
        problems.append(f"{n_pos_diff} no-effect positions differ from the null world")
    return {
        "n_core_rows": len(core_rows),
        "n_no_effect_rows": len(ne_rows),
        "pairs_1to1": len(set(paired)) == len(paired) == len(core_rows) and not missing,
        "prefix_identical": f"{n_prefix_same}/{len(ne_rows)}",
        "schedule_identical": f"{n_sched_same}/{len(ne_rows)}",
        "scored_no_effect_positions": n_pos,
        "positions_where_executions_differ": n_pos_diff,
        "problems": problems,
    }


def tv_distance(a: Counter, b: Counter) -> float:
    na, nb = sum(a.values()), sum(b.values())
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in keys)


def _prefix_group(row: dict) -> int:
    p = row["prefix_len"]
    return p if p in (5, 15, 35, 55) else p - 1


def _trajectory(row: dict) -> tuple:
    steps = row["trace_prefix"] + row["trace_original"]
    return row["domain"], json.dumps([[s["op"], s["args"]] for s in steps], sort_keys=True)


def coverage_audit(core_rows: list[dict], ne_rows: list[dict]) -> dict:
    axes = {
        "domain": lambda r: r["domain"],
        "prefix_group": _prefix_group,
        "trajectory": _trajectory,
        "intervention_kind": lambda r: r["intervention"]["kind"],
    }
    out = {"tvd": {}, "counts": {}}
    for name, fn in axes.items():
        a, b = Counter(map(fn, core_rows)), Counter(map(fn, ne_rows))
        out["tvd"][name] = round(tv_distance(a, b), 4)
        if name != "trajectory":
            out["counts"][name] = {
                "core_v1": {str(k): v for k, v in sorted(a.items())},
                "no_effect_v1": {str(k): v for k, v in sorted(b.items())},
            }
    out["counts"]["benign_tier"] = dict(sorted(Counter(r["benign_tier"] for r in ne_rows).items()))
    out["counts"]["contingency_class"] = dict(sorted(Counter(r["contingency_class"] for r in ne_rows).items()))
    out["counts"]["latent_divergence"] = sum(bool(r.get("latent_divergence")) for r in ne_rows)
    return out


# ---------------------------------------------------------------------------
# Grading

def _grade_step(step: dict, oracle) -> tuple[bool, bool, bool, bool]:
    """(value_match, status_match, well_formed, stored_flag_disagrees).

    Always re-grades `predicted`; `checked_steps` guarantees it is present.
    """
    vm = grade_value_match(step["predicted"], oracle)
    sm = grade_status_match(step["predicted"], oracle)
    v, w, s = bool(vm["match"]), bool(vm.get("well_formed")), bool(sm["match"])
    disagree = "value_match" in step and bool(step["value_match"]) != v
    return v, s, w, disagree


def _check_gt(step: dict, oracle) -> bool:
    return "ground_truth" not in step or canon(step["ground_truth"]) == canon(oracle)


def score_core(run: dict, core_rows: list[dict], budget: int) -> dict:
    catalog = {}
    for r in core_rows:
        key = schedule_key_from_catalog(r, budget)
        if key in catalog:
            raise ValueError("Core-v1 40-step schedules are not unique; cannot join")
        catalog[key] = r
    eff = {r["item_id"]: {j for j, (a, b) in enumerate(zip(r["trace_original"][:budget],
                                                          r["trace_counterfactual"][:budget]))
                          if canon(a["response"]) != canon(b["response"])}
           for r in core_rows}
    candidates: dict[str, list[dict]] = defaultdict(list)
    for run_row in run["per_item"]:
        if not run_row or "error" in run_row:
            continue
        row = catalog.get(schedule_key_from_run(run_row, budget))
        if row is not None:
            candidates[row["item_id"]].append(run_row)
    # Development runs can contain other interventions on the same trajectory
    # whose first 40 responses coincide with a Core-v1 row (only possible when
    # that row has no effect step in the scored window). Keep the run row that
    # carries the catalog item_id and cell; anything still ambiguous is an error.
    by_id = {r["item_id"]: r for r in core_rows}
    matched: dict[str, dict] = {}
    n_collisions = 0
    for iid, rows in candidates.items():
        if len(rows) > 1:
            n_collisions += 1
            kind = by_id[iid]["intervention"]["kind"]
            rows = [x for x in rows if x.get("item_id") == iid
                    and x.get("intervention_kind", kind) == kind]
            if len(rows) != 1:
                raise ValueError(f"Core run has an ambiguous match for catalog item {iid}")
        matched[iid] = rows[0]
    per_item = {}
    disagree = id_agree = 0
    for r in core_rows:
        run_row = matched.get(r["item_id"])
        if run_row is None:
            continue
        id_agree += run_row.get("item_id") == r["item_id"]
        steps = checked_steps(run_row, r, budget)
        rec = {"eff_ok": 0, "eff_n": 0, "all_ok": 0, "all_n": 0, "infra": 0, "wrong": 0, "null": 0}
        for s in steps:
            j = s["suffix_idx"]
            if is_infrastructure_failure(s, run_row):   # provider error, no model output: not scored
                rec["infra"] += 1
                continue
            v, _, _, d = _grade_step(s, r["trace_counterfactual"][j]["response"])
            disagree += d
            rec["all_ok"] += v
            rec["all_n"] += 1
            if j in eff[r["item_id"]]:
                rec["eff_ok"] += v
                rec["eff_n"] += 1
                if not v:   # null retreat: the wrong prediction is the no-intervention response
                    rec["wrong"] += 1
                    rec["null"] += bool(grade_value_match(s["predicted"], r["trace_original"][j]["response"])["match"])
        rec["complete"] = ({s["suffix_idx"] for s in steps}
                           == set(range(min(budget, len(r["trace_counterfactual"])))))
        per_item[r["item_id"]] = rec
    unmatched_ids = sum(1 for x in run["per_item"]
                        if x and "error" not in x and x.get("item_id") in by_id
                        and x.get("item_id") not in matched
                        and x.get("intervention_kind", by_id[x["item_id"]]["intervention"]["kind"])
                        == by_id[x["item_id"]]["intervention"]["kind"])
    return {"per_item": per_item, "n_matched": len(matched),
            "infra_excluded": sum(v["infra"] for v in per_item.values()),
            "n_rows_with_catalog_id_but_no_schedule_match": unmatched_ids,
            "n_item_id_agrees": id_agree, "stored_flag_disagreements": disagree,
            "n_schedule_collisions_resolved": n_collisions}


def score_no_effect(run: dict, ne_rows: list[dict], budget: int) -> dict:
    by_id = {r["item_id"]: r for r in ne_rows}
    seen = Counter(x.get("item_id") for x in run["per_item"] if x and "error" not in x)
    dup = [k for k, v in seen.items() if v > 1 and k in by_id]
    if dup:
        raise ValueError(f"no-effect run has duplicate rows for {len(dup)} item_ids")
    per_item = {}
    disagree = gt_bad = 0
    for run_row in run["per_item"]:
        if not run_row or "error" in run_row:
            continue
        r = by_id.get(run_row.get("item_id"))
        if r is None:
            continue
        steps = checked_steps(run_row, r, budget)
        rec = {"ok": 0, "n": 0, "wf": 0, "sm": 0, "over": 0, "infra": 0}
        for s in steps:
            oracle = r["trace_counterfactual"][s["suffix_idx"]]["response"]
            gt_bad += not _check_gt(s, oracle)
            if is_infrastructure_failure(s, run_row):   # provider error, no model output: not scored
                rec["infra"] += 1
                continue
            v, sm, w, d = _grade_step(s, oracle)
            disagree += d
            rec["ok"] += v
            rec["n"] += 1
            rec["wf"] += w
            rec["sm"] += sm
            # Oracle == null world on this split, so a well-formed miss is a
            # predicted change that does not happen.
            rec["over"] += w and not v
        rec["complete"] = ({s["suffix_idx"] for s in steps}
                           == set(range(min(budget, len(r["trace_counterfactual"])))))
        per_item[r["item_id"]] = rec
    return {"per_item": per_item, "ground_truth_mismatches": gt_bad,
            "stored_flag_disagreements": disagree,
            "infra_excluded": sum(v["infra"] for v in per_item.values())}


# ---------------------------------------------------------------------------
# Statistics

def ratio(pairs: list[tuple[int, int]]) -> float | None:
    den = sum(p[1] for p in pairs)
    return 100.0 * sum(p[0] for p in pairs) / den if den else None


def bootstrap_ci(pairs: list[tuple[int, int]], n_boot: int, seed: int):
    if not pairs or n_boot <= 0:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        den = sum(p[1] for p in sample)
        vals.append(100.0 * sum(p[0] for p in sample) / den if den else 0.0)
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]]


def summarize(pairs, n_boot, seed) -> dict:
    return {
        "value": ratio(pairs),
        "ci95": bootstrap_ci(pairs, n_boot, seed),
        "num": sum(p[0] for p in pairs),
        "den": sum(p[1] for p in pairs),
        "n_scenarios": len(pairs),
    }


def score_pair(label: str, core_run_path: Path, ne_run_path: Path,
               core_rows: list[dict], ne_rows: list[dict], budget: int,
               n_boot: int, seed: int, allow_partial: bool,
               allow_mixed: bool = False, catalog_shas: tuple = (None, None)) -> dict:
    core_run, ne_run = load_run(core_run_path), load_run(ne_run_path)
    # Run provenance: both outputs must come from one run configuration each, and
    # the two must agree; reported model / protocol / prompt come from row stamps.
    d_core = rp.describe_run(core_run, str(core_run_path))
    d_ne = rp.describe_run(ne_run, str(ne_run_path))
    for d, run, rows, name in ((d_core, core_run, core_rows, "the Core-v1 catalog"),
                               (d_ne, ne_run, ne_rows, "the no-effect-v1 catalog")):
        if d["status"] in (rp.STAMPED, rp.MIXED):  # stamped rows: join by catalog-row identity
            rp.bind_to_catalog(d, run, rows, None, budget=budget, catalog_name=name,
                               require_complete=not allow_partial)
    prov = rp.check([d_core, d_ne], allow_mixed=allow_mixed, tool=f"score_paired [{label}]",
                    extra_problems=rp.combination_problems([d_core, d_ne])
                    + rp.catalog_problems(d_core, catalog_shas[0], "the Core-v1 catalog")
                    + rp.catalog_problems(d_ne, catalog_shas[1], "the no-effect-v1 catalog"))
    # Both runs must be the same model (rp.MODEL_ALIASES), legacy or stamped, and a
    # label that names a known model must name that one.
    ids = {side: rp.model_identity(rp.report_fields(d)["model"]) for side, d in (("core", d_core), ("ne", d_ne))}
    if ids["core"] != ids["ne"]:
        raise SystemExit(f"error: [{label}] the Core-v1 run is {ids['core']} ({rp.report_fields(d_core)['model']}) "
                         f"but the no-effect run is {ids['ne']} ({rp.report_fields(d_ne)['model']}); "
                         "a pair must be one model")
    if label in rp.MODEL_ALIASES and label != ids["core"]:
        raise SystemExit(f"error: label {label!r} names another model than the runs ({ids['core']})")
    c = score_core(core_run, core_rows, budget)
    n = score_no_effect(ne_run, ne_rows, budget)

    pairs_complete = [r for r in ne_rows
                      if r["paired_item_id"] in c["per_item"] and r["item_id"] in n["per_item"]]
    incomplete_items = (sum(not v["complete"] for v in c["per_item"].values())
                        + sum(not v["complete"] for v in n["per_item"].values()))
    complete = (len(pairs_complete) == len(ne_rows) == len(core_rows)
                and incomplete_items == 0 and n["ground_truth_mismatches"] == 0)
    if not complete and not allow_partial:
        raise SystemExit(
            f"[{label}] incomplete or inconsistent pair: {len(pairs_complete)}/{len(ne_rows)} "
            f"pairs present, {incomplete_items} items with fewer than {budget} scored steps, "
            f"{n['ground_truth_mismatches']} no-effect ground-truth mismatches, "
            f"{c['n_rows_with_catalog_id_but_no_schedule_match']} Core run rows that carry a "
            "Core-v1 item_id but do not match its 40-call schedule. "
            "Pass --allow-partial to score anyway (the output is then marked complete=false).")

    # Score only scenarios whose pair is present on both sides, in catalog order.
    keep_ne = [r["item_id"] for r in pairs_complete]
    keep_core = [r["paired_item_id"] for r in pairs_complete]
    keep_core_set = set(keep_core)
    core_order = [r["item_id"] for r in core_rows if r["item_id"] in keep_core_set]

    eff_pairs = [(c["per_item"][i]["eff_ok"], c["per_item"][i]["eff_n"])
                 for i in core_order if c["per_item"][i]["eff_n"]]
    all_pairs = [(c["per_item"][i]["all_ok"], c["per_item"][i]["all_n"]) for i in core_order]
    ne_pairs = [(n["per_item"][i]["ok"], n["per_item"][i]["n"]) for i in keep_ne]
    over = sum(n["per_item"][i]["over"] for i in keep_ne)
    wf = sum(n["per_item"][i]["wf"] for i in keep_ne)
    sm = sum(n["per_item"][i]["sm"] for i in keep_ne)
    n_pos = sum(n["per_item"][i]["n"] for i in keep_ne)
    over_pairs = [(n["per_item"][i]["over"], n["per_item"][i]["wf"]) for i in keep_ne
                  if n["per_item"][i]["wf"]]

    ne_meta = {r["item_id"]: r for r in ne_rows}
    strata = {}
    for field in ("contingency_class", "benign_tier"):
        groups = defaultdict(list)
        for i in keep_ne:
            groups[ne_meta[i][field]].append(i)
        strata[field] = {
            g: {
                "no_effect_vm": summarize([(n["per_item"][i]["ok"], n["per_item"][i]["n"]) for i in ids],
                                          n_boot, seed),
                "over_prediction_rate": ratio([(n["per_item"][i]["over"], n["per_item"][i]["wf"]) for i in ids]),
            }
            for g, ids in sorted(groups.items())
        }

    stamped_report = rp.needs_report(prov)

    def meta(run, desc):
        out = {k: run.get(k) for k in ("model", "strategy", "window", "thinking", "suffix_budget", "n_items")}
        if stamped_report:  # take model / protocol / prompt version from the row stamps
            out.update(rp.report_fields(desc))
            out["provenance"] = desc["status"]
        return out

    core_meta, ne_meta_out = meta(core_run, d_core), meta(ne_run, d_ne)
    protocol_keys = ("strategy", "window", "thinking", "suffix_budget")
    if stamped_report:
        protocol_keys = protocol_keys + ("prompt_version",)
    mismatched = [k for k in protocol_keys if core_meta.get(k) != ne_meta_out.get(k)]

    extra = {}
    if stamped_report:
        extra["provenance"] = prov
        if prov["mixed"]:
            extra["mixed_run_provenance"] = True
            extra["warning"] = rp.warning_line(prov)

    return {
        "label": label,
        "complete": complete,
        **extra,
        "core_run": {"path": str(core_run_path), **core_meta},
        "no_effect_run": {"path": str(ne_run_path), **ne_meta_out},
        "protocol_mismatch": mismatched,
        "pairs_scored": f"{len(pairs_complete)}/{len(ne_rows)}",
        "effect_step_vm": summarize(eff_pairs, n_boot, seed),
        "null_retreat": {
            **summarize([(c["per_item"][i]["null"], c["per_item"][i]["wrong"]) for i in core_order
                         if c["per_item"][i]["wrong"]], n_boot, seed),
            "definition": "wrong Core-v1 effect-step predictions (scored steps; calls with a provider error "
                          "excluded) that equal the no-intervention response (trace_original) under the "
                          "grader's value-match test, as a percentage of all wrong effect-step predictions",
        },
        "no_effect_vm": summarize(ne_pairs, n_boot, seed),
        "no_effect_over_prediction_rate": {
            "value": ratio(over_pairs), "ci95": bootstrap_ci(over_pairs, n_boot, seed),
            "num": over, "den": wf,
            "definition": "well-formed no-effect predictions that differ from the "
                          "(unchanged) oracle response, as a share of well-formed predictions",
        },
        "no_effect_sm": 100.0 * sm / n_pos if n_pos else None,
        "no_effect_well_formed": 100.0 * wf / n_pos if n_pos else None,
        "no_effect_strata": strata,
        "infrastructure_failures_excluded": {
            "core_run_steps": sum(c["per_item"][i]["infra"] for i in core_order),
            "no_effect_run_steps": sum(n["per_item"][i]["infra"] for i in keep_ne),
            "rule": INFRA_RULE,
        },
        "diagnostics": {
            "core_all_call_vm": ratio(all_pairs),
            "core_items_matched": c["n_matched"],
            "core_matched_rows_with_same_item_id": c["n_item_id_agrees"],
            "core_stored_flag_disagreements": c["stored_flag_disagreements"],
            "core_schedule_collisions_resolved_by_item_id": c["n_schedule_collisions_resolved"],
            "core_rows_with_catalog_id_but_no_schedule_match": c["n_rows_with_catalog_id_but_no_schedule_match"],
            "no_effect_items_matched": len(n["per_item"]),
            "no_effect_ground_truth_mismatches": n["ground_truth_mismatches"],
            "no_effect_stored_flag_disagreements": n["stored_flag_disagreements"],
            "items_with_missing_steps": incomplete_items,
        },
    }


INFRA_RULE = ("calls that returned a provider error without any model output are excluded "
              "from every numerator and denominator (grader.is_infrastructure_failure)")


def fmt(x, nd=1):
    return "--" if x is None else f"{x:.{nd}f}"


def fmt_ci(s):
    ci = s.get("ci95")
    return f"{fmt(s['value'])} [{fmt(ci[0])}, {fmt(ci[1])}]" if ci else fmt(s["value"])


def print_table(results: list[dict]) -> None:
    print("Paired report: (Core-v1 effect-step VM, no-effect-v1 VM). Report BOTH; do not collapse.")
    head = ("model", "Eff VM [95% CI]", "NE VM [95% CI]", "over-pred %",
            "NE VM contingent", "NE VM global_no_op", "complete")
    print(" | ".join(head))
    for r in results:
        cc = r["no_effect_strata"]["contingency_class"]
        print(" | ".join([
            r["label"], fmt_ci(r["effect_step_vm"]), fmt_ci(r["no_effect_vm"]),
            fmt(r["no_effect_over_prediction_rate"]["value"]),
            fmt(cc.get("contingent", {}).get("no_effect_vm", {}).get("value")),
            fmt(cc.get("global_no_op", {}).get("no_effect_vm", {}).get("value")),
            str(r["complete"]),
        ]))
        if r["protocol_mismatch"]:
            print(f"  WARNING: protocol fields differ between the two runs: {r['protocol_mismatch']}")
        if r.get("mixed_run_provenance"):
            print(f"  {r['warning']}")
        elif r.get("provenance"):
            c = r["core_run"]
            print(f"  provenance: stamped rows, model={c.get('model')} protocol={c.get('strategy')} "
                  f"window={c.get('window')} prompt={c.get('prompt_version')}")
    print("Reference points: null-world predictor = (Eff VM 0.0, NE VM 100.0).")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--core-run", type=Path, help="model output on counterfactuals/core_v1.jsonl")
    ap.add_argument("--no-effect-run", type=Path, help="model output on counterfactuals/no_effect_v1.jsonl")
    ap.add_argument("--label", default=None, help="model label for --core-run/--no-effect-run")
    ap.add_argument("--pair", nargs=3, action="append", default=[],
                    metavar=("LABEL", "CORE_RUN", "NO_EFFECT_RUN"))
    ap.add_argument("--core-jsonl", type=Path, default=DEFAULT_CORE)
    ap.add_argument("--no-effect-jsonl", type=Path, default=DEFAULT_NO_EFFECT)
    ap.add_argument("--suffix-budget", type=int, default=SUFFIX_BUDGET)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--allow-partial", action="store_true",
                    help="score incomplete pairs; output is marked complete=false")
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    jobs = [(lab, Path(c), Path(n)) for lab, c, n in args.pair]
    if args.core_run or args.no_effect_run:
        if not (args.core_run and args.no_effect_run):
            ap.error("--core-run and --no-effect-run must be given together")
        jobs.append((args.label or args.core_run.stem, args.core_run, args.no_effect_run))
    if not jobs:
        ap.error("give --core-run/--no-effect-run or at least one --pair")
    dup = sorted(lab for lab, n in Counter(lab for lab, _, _ in jobs).items() if n > 1)
    if dup:
        ap.error(f"duplicate pair label(s) {', '.join(map(repr, dup))}: every pair needs its own label")

    core_rows = load_jsonl(args.core_jsonl)
    ne_rows = load_jsonl(args.no_effect_jsonl)
    pairing = validate_pairing(core_rows, ne_rows, args.suffix_budget)
    if pairing["problems"]:
        raise SystemExit(f"catalog pairing check failed: {pairing['problems']}")

    try:
        shas = (rp.file_sha256(args.core_jsonl), rp.file_sha256(args.no_effect_jsonl))
        results = [score_pair(lab, c, n, core_rows, ne_rows, args.suffix_budget,
                              args.bootstrap, args.seed, args.allow_partial,
                              allow_mixed=args.allow_mixed, catalog_shas=shas)
                   for lab, c, n in jobs]
    except InvalidRunError as e:
        raise SystemExit(f"invalid run file: {e}")
    except rp.MixedRunError as e:
        raise SystemExit(f"error: {e}")
    print_table(results)

    if args.json:
        out = {
            "reporting_rule": REPORTING_RULE,
            "protocol_note": "Both runs of a pair must use the same protocol "
                             "(e.g. answers revealed = --strategy sliding --window 20).",
            "catalogs": {"core_v1": str(args.core_jsonl), "no_effect_v1": str(args.no_effect_jsonl)},
            "suffix_budget": args.suffix_budget,
            "bootstrap": {"B": args.bootstrap, "seed": args.seed, "unit": "scenario",
                          "estimator": "ratio of sums", "interval": "percentile 2.5/97.5",
                          "order": "catalog order"},
            "pairing_audit": pairing,
            "coverage_audit": coverage_audit(core_rows, ne_rows),
            "reference_points": {"null_world_predictor": {"effect_step_vm": 0.0, "no_effect_vm": 100.0}},
            "models": {r["label"]: r for r in results},
        }
        mixed = [r["label"] for r in results if r.get("mixed_run_provenance")]
        if mixed:
            out["mixed_run_provenance"] = mixed
            out["warning"] = ("MIXED RUN PROVENANCE in " + ", ".join(mixed) +
                              "; see models[...].provenance")
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
