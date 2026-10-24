"""Non-simulating reference policies for every memory protocol (CPU only, no model).

Each policy receives exactly the information a model receives under the named
protocol and performs no program simulation. Its score is a reference point
for reading a model score under the same protocol:

  answers revealed (tf1step_sliding_k20)
    replay_without_intervention  predict the response of the unpatched service
    copy_last_same_op            repeat the most recent visible response of the same operation
    copy_last_same_call          repeat the most recent visible response of the same call
                                 (operation and arguments)
  no feedback (no_suffix_feedback) and free rollout (free_rollout)
    copy_last_same_op, copy_last_same_call; in free rollout the history holds
    the policy's own earlier predictions, as it holds a model's.

"Visible" is exactly the call history of the evaluator's prompt
(`scripts/eval_llm_counterfactual.py`): with answers revealed and in free
rollout, the most recent 20 calls of the observed calls followed by the future
calls (correct responses or the policy's own predictions); with no feedback,
all observed calls. With no matching visible call the policy predicts `{}`. Scores use the released grader on the first 40
future calls; intervals resample scenarios (B=2000, seed 17, ratio of sums,
catalog order), as in `scripts/score_paired.py`.

Usage:
    python scripts/reference_policies.py --core-jsonl counterfactuals/core_v1.jsonl --json scratch/reference_policies.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_paired as sp  # noqa: E402
from grader import grade_status_match, grade_value_match  # noqa: E402

BUDGET = 40
WINDOW = 20
from core_v1 import CORE_V1_SHA256  # noqa: E402


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def load_items(path: Path) -> list[dict]:
    items = []
    for row in sp.load_jsonl(path):
        orig = [s["response"] for s in row["trace_original"]][:BUDGET]
        cf = [s["response"] for s in row["trace_counterfactual"]][:BUDGET]
        calls = [(s["op"], canon(s["args"])) for s in row["trace_counterfactual"]][:BUDGET]
        prefix = [(s["op"], canon(s["args"]), s["response"]) for s in row["trace_prefix"]]
        items.append({"orig": orig, "cf": cf, "calls": calls, "prefix": prefix,
                      "effect": [i for i in range(len(cf)) if canon(orig[i]) != canon(cf[i])]})
    return items


def copy_last(item: dict, *, keyed: bool, history_source: str) -> list:
    """history_source: 'oracle' (answers revealed), 'none' (no feedback), 'self' (free rollout)."""
    key_of = (lambda op, args: (op, args)) if keyed else (lambda op, args: op)
    history = [(key_of(op, args), resp) for op, args, resp in item["prefix"]]
    out = []
    for i, (op, args) in enumerate(item["calls"]):
        key = key_of(op, args)
        # Same window as the prompt: sliding / free_rollout keep the last WINDOW
        # calls of prefix + future history; prefix_only shows the whole prefix.
        visible = history if history_source == "none" else history[-WINDOW:]
        pred = {}
        for seen_key, seen_resp in reversed(visible):
            if seen_key == key:
                pred = seen_resp
                break
        out.append(pred)
        if history_source == "oracle":
            history.append((key, item["cf"][i]))
        elif history_source == "self":
            history.append((key, pred))
    return out


POLICIES = {
    "tf1step_sliding_k20": {
        "replay_without_intervention": lambda it: it["orig"],
        "copy_last_same_op": lambda it: copy_last(it, keyed=False, history_source="oracle"),
        "copy_last_same_call": lambda it: copy_last(it, keyed=True, history_source="oracle"),
    },
    "no_suffix_feedback": {
        "copy_last_same_op": lambda it: copy_last(it, keyed=False, history_source="none"),
        "copy_last_same_call": lambda it: copy_last(it, keyed=True, history_source="none"),
    },
    "free_rollout": {
        "copy_last_same_op": lambda it: copy_last(it, keyed=False, history_source="self"),
        "copy_last_same_call": lambda it: copy_last(it, keyed=True, history_source="self"),
    },
}


def score(items: list[dict], policy, n_boot: int, seed: int) -> tuple[dict, list]:
    per, preds_all = [], []
    for it in items:
        preds = policy(it)
        preds_all.append(preds)
        vm = [bool(grade_value_match(p, t)["match"]) for p, t in zip(preds, it["cf"])]
        sm = [bool(grade_status_match(p, t)["match"]) for p, t in zip(preds, it["cf"])]
        per.append((sum(vm[i] for i in it["effect"]), len(it["effect"]), sum(vm), len(vm), sum(sm)))

    def stat(a: int, b: int) -> dict:
        pairs = [(p[a], p[b]) for p in per if p[b]]
        num, den = sum(x for x, _ in pairs), sum(y for _, y in pairs)
        lo, hi = sp.bootstrap_ci(pairs, n_boot, seed)
        return {"value": 100.0 * num / den, "ci95": [lo, hi], "half_width": (hi - lo) / 2,
                "num": num, "den": den}
    return {"effect_step_vm": stat(0, 1), "all_call_vm": stat(2, 3), "sm": stat(4, 3)}, preds_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-jsonl", type=Path, default=Path("counterfactuals/core_v1.jsonl"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--allow-mixed", action="store_true",
                    help="score a catalog other than Core-v1 (the output is then labelled with its sha256)")
    args = ap.parse_args()
    import run_provenance as rp
    sha = rp.file_sha256(args.core_jsonl)
    if sha != CORE_V1_SHA256 and not args.allow_mixed:
        raise SystemExit(f"error: {args.core_jsonl} is not the Core-v1 catalog (sha256 {sha}); "
                         "the reference policies are defined on Core-v1. Pass --allow-mixed to score it anyway.")
    items = load_items(args.core_jsonl)
    out = {"catalog": args.core_jsonl.name,
           **({"catalog_sha256": sha, "warning": "WARNING: catalog other than Core-v1 (--allow-mixed)"}
              if sha != CORE_V1_SHA256 else {}),
           "scenarios": len(items),
           "effect_steps": sum(len(i["effect"]) for i in items), "suffix_budget": BUDGET,
           "window": WINDOW,
           "bootstrap": f"scenario resampling, B={args.bootstrap}, seed {args.seed}, catalog order, ratio of sums",
           "protocols": {}}
    for proto, policies in POLICIES.items():
        out["protocols"][proto] = {}
        for name, fn in policies.items():
            res, _ = score(items, fn, args.bootstrap, args.seed)
            out["protocols"][proto][name] = res
            e, v, m = res["effect_step_vm"], res["all_call_vm"], res["sm"]
            print(f"{proto:20s} {name:28s} Eff VM {e['value']:6.2f} +/- {e['half_width']:.2f}   "
                  f"VM {v['value']:6.2f} +/- {v['half_width']:.2f}   SM {m['value']:6.2f} +/- {m['half_width']:.2f}")
    out["visibility"] = ("history as in the evaluator's prompt: last 20 calls of observed + future calls "
                         "(tf1step_sliding_k20, free_rollout); all observed calls (no_suffix_feedback)")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
