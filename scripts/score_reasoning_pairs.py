"""Reasoning off vs on for DeepSeek V4-Flash on Core-v1 (same provider, same harness).

Inputs (dataset repository):
  reference_results/raw/deepseek_v4_flash_{tf1step_sliding_k20,no_suffix_feedback}.json
      reasoning off (`deepseek-chat`)
  reference_results/reasoning_raw/deepseek_v4_flash_thinking_{tf1step_sliding_k20,no_suffix_feedback}.json
      reasoning on (`deepseek-reasoner`, reasoning effort high, 12,000-token budget)

For each protocol, both runs are scored on the scenarios present in the
reasoning-on file (its `coverage` block lists any others), with the validated
joins and re-grading of `scripts/audit_reference_results.py`: effect-step VM,
all-call VM, SM, and the share of parseable responses, each with a scenario
bootstrap half-width (B=2000, seed 17, catalog order).

Usage:
    python scripts/score_reasoning_pairs.py --dataset-root path/to/cte-bench-core-v1 --json scratch/reasoning.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import audit_reference_results as ar  # noqa: E402
import run_provenance as rp  # noqa: E402
import score_paired as sp  # noqa: E402

# The pair differs by design in model id (deepseek-chat / deepseek-reasoner),
# reasoning, and token budget; everything else must agree.
PAIR_FIELDS = ("strategy", "window", "prompt_version", "suffix_budget", "explain", "state")
# The study's expected settings (legacy files are checked on the fields they declare).
STUDY = "score_reasoning_pairs (DeepSeek V4-Flash reasoning off/on)"
EXPECT_PROTOCOL = {"tf1step_sliding_k20": {"strategy": "sliding", "window": 20, "suffix_budget": 40},
                   "no_suffix_feedback": {"strategy": "prefix_only", "suffix_budget": 40}}
EXPECT_SIDE = {"off": {"thinking": "off", "model": ("DS-V4-Lite/sliding", "deepseek/deepseek-chat")},
               "on": {"thinking": "on", "model": ("deepseek/deepseek-reasoner",)}}

PROTOCOLS = ("tf1step_sliding_k20", "no_suffix_feedback")
KEEP = ("model", "scenarios_covered", "effect_steps", "effect_step_vm", "all_call_vm", "sm",
        "ci95_half_width", "outcomes_pct_of_calls", "scored_calls", "missing_scenarios",
        "invalid_rows", "incomplete_rows", "rows_answering_other_scenario", "duplicate_rows_except_id",
        "infrastructure_failures_excluded")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    args = ap.parse_args()
    root = args.dataset_root
    core = sp.load_jsonl(root / "counterfactuals" / "core_v1.jsonl")
    out = {"model": "DeepSeek V4-Flash", "protocols": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for proto in PROTOCOLS:
            on_path = root / "reference_results" / "reasoning_raw" / f"deepseek_v4_flash_thinking_{proto}.json"
            off_path = root / "reference_results" / "raw" / f"deepseek_v4_flash_{proto}.json"
            on = json.loads(on_path.read_text())
            ids = {r["item_id"] for r in on["per_item"]}
            off = json.loads(off_path.read_text())
            off["per_item"] = [r for r in off["per_item"] if r["item_id"] in ids]
            if len(off["per_item"]) != len(ids):
                raise SystemExit(f"{off_path.name}: {len(off['per_item'])} rows for {len(ids)} scenarios")
            off_tmp = Path(tmp) / f"off_{proto}.json"
            off_tmp.write_text(json.dumps(off))
            descs = [rp.describe_run(off, str(off_path)), rp.describe_run(on, str(on_path))]
            for d, side in zip(descs, ("off", "on")):
                rp.expect_config(d, {**EXPECT_PROTOCOL[proto], **EXPECT_SIDE[side]}, STUDY)
            try:
                prov = rp.check(descs, allow_mixed=args.allow_mixed, tool=f"score_reasoning_pairs [{proto}]",
                                extra_problems=rp.combination_problems(descs, fields=PAIR_FIELDS))
            except rp.MixedRunError as e:
                raise SystemExit(f"error: {e}")
            res = {}
            for side, path in (("off", off_tmp), ("on", on_path)):
                try:
                    a = ar.audit_file(path, core, "schedule", allow_mixed=args.allow_mixed,
                                      catalog_path=root / "counterfactuals" / "core_v1.jsonl")
                except (rp.MixedRunError, sp.InvalidRunError) as e:
                    raise SystemExit(f"error: {e}")
                res[side] = {k: a[k] for k in KEEP}
                for bad in ("invalid_rows", "incomplete_rows", "rows_answering_other_scenario",
                            "duplicate_rows_except_id"):
                    if res[side][bad]:
                        raise SystemExit(f"{proto}/{side}: {bad} = {res[side][bad]}")
            if prov["mixed"]:
                out["model"] = "MIXED (see protocols[...].provenance)"
            if rp.needs_report(prov):
                res["provenance"] = prov
                for side, d in zip(("off", "on"), descs):
                    res[side].update(rp.report_fields(d))
                if prov["mixed"]:
                    res["warning"] = rp.warning_line(prov)
                    print(f"{proto}: {res['warning']}")
            res["scenarios"] = len(ids)
            res["reasoning_on_settings"] = {k: on.get(k) for k in ("model", "thinking", "reasoning_effort", "max_tokens")}
            res["files"] = {"off": f"reference_results/raw/{off_path.name}",
                            "on": f"reference_results/reasoning_raw/{on_path.name}"}
            out["protocols"][proto] = res
            for side in ("off", "on"):
                s = res[side]
                h = s["ci95_half_width"]
                print(f"{proto:20s} {side:3s} scenarios {s['scenarios_covered']:3d} effect steps {s['effect_steps']:4d} "
                      f"Eff VM {s['effect_step_vm']:6.2f} +/- {h['effect_step_vm']:.2f}  "
                      f"VM {s['all_call_vm']:6.2f} +/- {h['all_call_vm']:.2f}  SM {s['sm']:6.2f} +/- {h['sm']:.2f}  "
                      f"parseable {s['outcomes_pct_of_calls']['well_formed']:.2f}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
