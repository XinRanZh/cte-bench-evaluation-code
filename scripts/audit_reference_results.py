"""Audit every released reference-output file against the released catalogs.

For each file under `reference_results/raw/` (Core-v1), `reference_results/
no_effect_v1_raw/` (no-effect-v1), `reference_results/adv24_raw/` (adv24
probe), `reference_results/reasoning_raw/` (Core-v1) and
`reference_results/renamed_raw/` (the 210 source-edit Core-v1 scenarios or
the renamed catalog) this
reports: rows, distinct item_ids, `__N`-suffixed ids, rows that match a
catalog scenario by the validated join, catalog scenarios with no matching
row, rows whose stored item_id disagrees with the scenario they actually
answer, pairs of rows identical except for item_id, scored calls, the
re-graded metrics (effect-step VM, all-call VM, SM, parse failures), and the
calls excluded as infrastructure failures (`infrastructure_failures_excluded`:
a provider error without any model output is not scored; see
`grader.is_infrastructure_failure`).

Joins (same rules as `scripts/score_paired.py`):
  * Core-v1 and adv24: by the 40-call (op, args, ground truth) schedule, which
    identifies a scenario uniquely; ties between rows of the same schedule are
    resolved by item_id, anything else is reported as ambiguous.
  * no-effect-v1: by item_id (schedules are not unique there), with every
    step's call checked against the catalog.
Every joined row's steps go through `score_paired.checked_steps` (each index
once, call equals catalog, `predicted` present) and are re-graded with the
released grader; stored flags are only compared.

Usage:
    python3 scripts/audit_reference_results.py --dataset-root <hf dataset dir> [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_provenance as rp  # noqa: E402
import score_paired as sp  # noqa: E402
from grader import (grade_status_match, grade_value_match, infrastructure_failures,  # noqa: E402
                    is_infrastructure_failure)

BUDGET = 40


def canon(o) -> str:
    return json.dumps(o, sort_keys=True)


def strip_id(row: dict) -> str:
    r = dict(row)
    r.pop("item_id", None)
    return canon(r)


def audit_file(path: Path, catalog: list[dict], join: str, allow_mixed: bool = False,
               catalog_path: Path | None = None) -> dict:
    run = sp.load_run(path)
    desc = rp.describe_run(run, str(path))
    if desc["status"] in (rp.STAMPED, rp.MIXED):  # stamped rows: identity, catalog hash, steps
        rp.bind_to_catalog(desc, run, catalog, catalog_path, budget=BUDGET, require_complete=False)
    prov = rp.check([desc], allow_mixed=allow_mixed, tool="audit_reference_results")
    rows = [r for r in run["per_item"] if r and "error" not in r]
    ids = [r.get("item_id") for r in rows]
    # Scenario key: item_id when unique in the catalog, else item_id#position
    # (the adv24 catalog reuses item_ids for variants).
    id_counts = Counter(c["item_id"] for c in catalog)
    uid = {i: (c["item_id"] if id_counts[c["item_id"]] == 1 else f"{c['item_id']}#{i}")
           for i, c in enumerate(catalog)}
    by_id = {uid[i]: c for i, c in enumerate(catalog)}
    eff = {uid[i]: {j for j, (a, b) in enumerate(zip(c["trace_original"][:BUDGET],
                                                    c["trace_counterfactual"][:BUDGET]))
                    if canon(a["response"]) != canon(b["response"])} for i, c in enumerate(catalog)}
    matched: dict[str, dict] = {}
    unmatched_rows, ambiguous, id_disagree = [], [], []
    if join == "schedule":
        keyed = defaultdict(list)
        for i, c in enumerate(catalog):
            keyed[sp.schedule_key_from_catalog(c, BUDGET)].append(uid[i])
        cand = defaultdict(list)
        for r in rows:
            cs = keyed.get(sp.schedule_key_from_run(r, BUDGET))
            if not cs:
                unmatched_rows.append(r.get("item_id"))
                continue
            for u in cs:
                cand[u].append(r)
        for u, rs in cand.items():
            cid = by_id[u]["item_id"]
            if len(rs) > 1:
                same = [r for r in rs if r.get("item_id") == cid]
                if len(same) != 1:
                    ambiguous.append(u)
                    continue
                rs = same
            matched[u] = rs[0]
            if rs[0].get("item_id") != cid:
                id_disagree.append((rs[0].get("item_id"), u))
    else:
        for r in rows:
            c = by_id.get(r.get("item_id"))
            if c is None:
                unmatched_rows.append(r.get("item_id"))
            elif r["item_id"] in matched:
                ambiguous.append(r["item_id"])
            else:
                matched[r["item_id"]] = r
    # rows answering a scenario under a different id are not counted as that id
    missing = [u for u in by_id if u not in matched]
    dup_pairs = []
    seen = defaultdict(list)
    for r in rows:
        seen[strip_id(r)].append(r.get("item_id"))
    for v in seen.values():
        if len(v) > 1:
            dup_pairs.append(v)

    calls = eff_ok = eff_n = all_ok = sm_ok = parse_fail = incomplete = flag_dis = 0
    status_only = status_wrong = 0
    infra = {"phase_a_steps": 0, "phase_b_calls": 0, "phase_c_calls": 0, "rows_affected": 0}
    invalid = []
    per_scen = []  # (eff_ok, eff_n, all_ok, n, sm_ok) in catalog order, for the bootstrap
    for iid in [u for u in by_id if u in matched]:
        r = matched[iid]
        c = by_id[iid]
        try:
            steps = sp.checked_steps(r, c, BUDGET)
        except sp.InvalidRunError as e:
            invalid.append(str(e))
            continue
        if {s["suffix_idx"] for s in steps} != set(range(min(BUDGET, len(c["trace_counterfactual"])))):
            incomplete += 1
        e_ok = e_n = a_ok = m_ok = n_scored = 0
        failed = infrastructure_failures(r)
        for key, ph in (("phase_a_steps", "phase_a"), ("phase_b_calls", "phase_b"), ("phase_c_calls", "phase_c")):
            infra[key] += failed[ph]
        infra["rows_affected"] += any(failed.values())
        for s in steps:
            if is_infrastructure_failure(s, r):   # provider error, no model output: not scored
                continue
            n_scored += 1
            oracle = c["trace_counterfactual"][s["suffix_idx"]]["response"]
            vm = grade_value_match(s["predicted"], oracle)
            v = bool(vm["match"])
            w = bool(vm.get("well_formed"))
            m = bool(grade_status_match(s["predicted"], oracle)["match"])
            calls += 1
            all_ok += v
            sm_ok += m
            a_ok += v
            m_ok += m
            parse_fail += not w
            status_only += w and m and not v
            status_wrong += w and not m
            flag_dis += "value_match" in s and bool(s["value_match"]) != v
            if s["suffix_idx"] in eff[iid]:
                eff_n += 1
                eff_ok += v
                e_n += 1
                e_ok += v
        per_scen.append((e_ok, e_n, a_ok, n_scored, m_ok))
    billing_calls = (run.get("billing_total") or {}).get("n_calls")
    pct = lambda a, b: round(100.0 * a / b, 4) if b else None  # noqa: E731

    def half(pairs):
        ci = sp.bootstrap_ci([p for p in pairs if p[1]], 2000, 17)
        return round((ci[1] - ci[0]) / 2, 4) if ci else None
    extra = {}
    reported = {"model": run.get("model"), "strategy": run.get("strategy")}
    if rp.needs_report(prov):  # stamped or mixed: report the row-stamp configuration
        fields = rp.report_fields(desc)
        reported = {"model": fields["model"], "strategy": fields["strategy"]}
        extra = {"prompt_version": fields["prompt_version"], "window": fields["window"],
                 "provenance": prov}
        if prov["mixed"]:
            extra["warning"] = rp.warning_line(prov)
    return {
        "file": path.as_posix(),
        **reported, **extra,
        "rows": len(rows), "distinct_item_ids": len(set(ids)),
        "suffixed_ids": sorted(i for i in ids if i and "__" in i),
        "catalog_scenarios": len(catalog), "scenarios_covered": len(matched),
        "missing_scenarios": missing, "unmatched_rows": unmatched_rows,
        "ambiguous": ambiguous, "rows_answering_other_scenario": id_disagree,
        "duplicate_rows_except_id": dup_pairs, "invalid_rows": invalid,
        "incomplete_rows": incomplete, "scored_calls": calls,
        "billing_n_calls": billing_calls, "stored_n_items": run.get("n_items"),
        "effect_steps": eff_n, "effect_step_vm": pct(eff_ok, eff_n),
        "all_call_vm": pct(all_ok, calls), "sm": pct(sm_ok, calls),
        "ci95_half_width": {"effect_step_vm": half([(p[0], p[1]) for p in per_scen]),
                            "all_call_vm": half([(p[2], p[3]) for p in per_scen]),
                            "sm": half([(p[4], p[3]) for p in per_scen]),
                            "bootstrap": "scenario resampling, B=2000, seed 17, catalog order, ratio of sums"},
        "outcomes_pct_of_calls": {"well_formed": pct(calls - parse_fail, calls), "parse_fail": pct(parse_fail, calls),
                                  "vm_exact": pct(all_ok, calls), "status_only": pct(status_only, calls),
                                  "status_wrong": pct(status_wrong, calls)},
        "stored_flag_disagreements": flag_dis,
        "infrastructure_failures_excluded": {**infra, "rule": sp.INFRA_RULE},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    args = ap.parse_args()
    root = args.dataset_root
    cat_paths = {
        "core": root / "counterfactuals" / "core_v1.jsonl",
        "ne": root / "counterfactuals" / "no_effect_v1.jsonl",
        "adv24": root / "counterfactuals" / "adv24_reasoning_probe.jsonl",
        "renamed": root / "counterfactuals" / "core_v1_renamed.jsonl",
    }
    cats = {k: sp.load_jsonl(v) for k, v in cat_paths.items()}
    # the renamed study's original arms ran on the 210 source-edit Core-v1 scenarios
    renamed_ids = {r["paired_item_id"] for r in cats["renamed"]}
    cats["core_source_edit"] = [c for c in cats["core"] if c["item_id"] in renamed_ids]
    cat_paths["core_source_edit"] = cat_paths["core"]
    # every released run; the renamed arms use the renamed catalog, the original arms Core-v1
    groups = [("raw", "core", "schedule"), ("no_effect_v1_raw", "ne", "item_id"),
              ("adv24_raw", "adv24", "schedule"), ("reasoning_raw", "core", "schedule"),
              ("renamed_raw", None, "schedule")]
    out = []
    for sub, group_cat, join in groups:
        for path in sorted((root / "reference_results" / sub).glob("*.json")):
            cat = group_cat or ("renamed" if "_renamed_" in path.name else "core_source_edit")
            try:
                res = audit_file(path, cats[cat], join, allow_mixed=args.allow_mixed,
                                 catalog_path=cat_paths[cat])
            except (rp.MixedRunError, sp.InvalidRunError) as e:
                raise SystemExit(f"error: {e}")
            if res.get("warning"):
                print(f"{path.name}: {res['warning']}")
            res["file"] = f"reference_results/{sub}/{path.name}"
            out.append(res)
    hdr = ("file", "rows", "ids", "__N", "covered", "missing", "other-id", "dup", "calls", "billing", "effN", "EffVM", "VM", "SM",
           "infra-excluded A/B/C")
    print(" | ".join(hdr))
    for r in out:
        print(" | ".join(str(x) for x in (
            r["file"].split("/", 1)[1], r["rows"], r["distinct_item_ids"], len(r["suffixed_ids"]),
            f"{r['scenarios_covered']}/{r['catalog_scenarios']}", len(r["missing_scenarios"]),
            len(r["rows_answering_other_scenario"]), len(r["duplicate_rows_except_id"]),
            r["scored_calls"], r["billing_n_calls"], r["effect_steps"], r["effect_step_vm"], r["all_call_vm"], r["sm"],
            "/".join(str(r["infrastructure_failures_excluded"][k])
                     for k in ("phase_a_steps", "phase_b_calls", "phase_c_calls")))))
        for k in ("missing_scenarios", "unmatched_rows", "ambiguous", "rows_answering_other_scenario",
                  "duplicate_rows_except_id", "invalid_rows"):
            if r[k]:
                print(f"    {k}: {r[k][:6]}")
        if r["incomplete_rows"] or r["stored_flag_disagreements"]:
            print(f"    incomplete_rows={r['incomplete_rows']} stored_flag_disagreements={r['stored_flag_disagreements']}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
