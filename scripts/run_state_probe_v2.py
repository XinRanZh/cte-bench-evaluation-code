"""Standalone runner for Phase C only, using the state-probe v2 grader.

Reads items from a counterfactuals JSONL (the authoritative catalog),
asks one LLM to produce the final hidden state per item, scores with
grade_state_probe_v2 (predeclared testbed paths + null-baseline
subtraction), streams every call to disk the moment it returns.

No Phase A, no Phase B. One call per item. Cheap to run on any model
we already ran Phase A+B for, so we can attach a v2 SP column to
headline results without re-running the whole eval.

Usage:
  python scripts/run_state_probe_v2.py \\
      --cf-jsonl /tmp/cte_stratified_200.jsonl \\
      --model deepseek/deepseek-v4-pro \\
      --thinking on --max-tokens 16000 --max-concurrency 32 \\
      --calls-dir results/full_grid_2026-04-29/ds_pro_200_sp_v2 \\
      --out results/full_grid_2026-04-29/ds_pro_200_sp_v2.json
"""
from __future__ import annotations
import argparse, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS.parent))

from eval_llm_counterfactual import (
    call_llm, make_state_prompt, SYSTEM_STATE,
)
from grader import parse_json_loose, grade_state_probe, grade_state_probe_v2
from testbeds import TESTBEDS


def _attach_source(item: dict) -> None:
    if item.get("source_code"):
        return
    mod = TESTBEDS[item["domain"]]
    item["source_code"] = mod.SOURCE_CODE


def probe_item(item: dict, model_id: str, max_tokens: int,
               thinking_mode: str, per_call_logger=None) -> dict:
    _attach_source(item)
    if item.get("final_state_cf") is None or item.get("final_state_original") is None:
        return {"item_id": item.get("item_id"),
                "domain": item.get("domain"),
                "error": "MISSING_GROUND_TRUTH_STATES"}

    domain_mod = TESTBEDS.get(item["domain"])
    probe_spec = getattr(domain_mod, "PROBE_PATHS_V2", None)
    if not probe_spec:
        return {"item_id": item.get("item_id"),
                "domain": item.get("domain"),
                "error": "NO_PROBE_SPEC_V2"}

    prompt = make_state_prompt(item, max_history=60)
    t0 = time.time()
    usage, cost, call_err, raw = {}, 0.0, None, ""
    try:
        raw, usage, cost = call_llm(model_id, SYSTEM_STATE, prompt, None,
                                    max_tokens=max_tokens,
                                    thinking_mode=thinking_mode)
        dt = time.time() - t0
        parsed = parse_json_loose(raw)
    except Exception as e:
        dt = time.time() - t0
        parsed = None
        call_err = str(e)[:200]

    probe_v1 = grade_state_probe(parsed, item["final_state_cf"]) if parsed else {"valid": False}
    probe_v2 = grade_state_probe_v2(parsed,
                                    item["final_state_cf"],
                                    item["final_state_original"],
                                    probe_spec) if parsed else {"valid": False}

    rec = {
        "item_id": item.get("item_id"),
        "domain": item.get("domain"),
        "intervention_kind": item["intervention"].get("kind"),
        "latency_s": round(dt, 2),
        "usage": usage,
        "cost_usd": cost,
        "raw_excerpt": raw[:200] if raw else "",
        "error": call_err,
        "probe_v1": probe_v1,
        "probe_v2": probe_v2,
    }

    if per_call_logger is not None:
        try:
            per_call_logger(rec)
        except Exception:
            pass
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-jsonl", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--thinking", choices=["off", "on"], default="off")
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--max-concurrency", type=int, default=8)
    ap.add_argument("--max-items", type=int, default=0,
                    help="0 = all items in catalog")
    ap.add_argument("--calls-dir", required=True,
                    help="Directory for streaming per-call jsonl output")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = []
    with open(args.cf_jsonl) as f:
        for line in f:
            items.append(json.loads(line))
            if args.max_items and len(items) >= args.max_items:
                break

    calls_dir = Path(args.calls_dir)
    calls_dir.mkdir(parents=True, exist_ok=True)
    calls_path = calls_dir / "calls.jsonl"

    # Resume support: read existing per-call log, skip items already done.
    done_ids = set()
    if calls_path.exists():
        with calls_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("item_id"):
                        done_ids.add(r["item_id"])
                except Exception:
                    continue
    if done_ids:
        print(f"[resume] {len(done_ids)} items already logged; skipping them")

    pending = [it for it in items if it.get("item_id") not in done_ids]
    print(f"evaluating {len(pending)}/{len(items)} items "
          f"with model={args.model} thinking={args.thinking}")

    calls_f = calls_path.open("a")
    lock = threading.Lock()

    def logger(rec):
        with lock:
            calls_f.write(json.dumps(rec, default=str) + "\n")
            calls_f.flush()
            try:
                os.fsync(calls_f.fileno())
            except OSError:
                pass

    total = {"n": 0, "cost": 0.0, "cache": 0, "prompt": 0}

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futs = {pool.submit(probe_item, it, args.model, args.max_tokens,
                            args.thinking, logger): it["item_id"]
                for it in pending}
        for fut in as_completed(futs):
            try:
                r = fut.result()
                total["n"] += 1
                total["cost"] += r.get("cost_usd", 0) or 0
                u = r.get("usage") or {}
                total["prompt"] += u.get("prompt_tokens", 0) or 0
                total["cache"] += u.get("cached_tokens", 0) or 0
                v2 = r.get("probe_v2") or {}
                v2_frac = v2.get("fraction_match")
                hit_pct = (total["cache"] / total["prompt"] * 100
                           if total["prompt"] else 0)
                print(f"  {r.get('item_id','?')[:8]:<8} "
                      f"[{r.get('domain',''):<12} {r.get('intervention_kind',''):<18}] "
                      f"v2={v2_frac if v2_frac is not None else '-'} "
                      f"neff={v2.get('n_effective','-')} "
                      f"dt={r['latency_s']}s ${r['cost_usd']:.4f} "
                      f"| run_total=${total['cost']:.2f} cache={hit_pct:.0f}%",
                      flush=True)
            except Exception as e:
                print(f"  FAILED: {e}", flush=True)

    calls_f.close()

    # Final aggregate dump
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with calls_path.open() as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    out_path.write_text(json.dumps({
        "model": args.model,
        "thinking": args.thinking,
        "n_records": len(records),
        "total_cost_usd": total["cost"],
        "cache_hit_pct": (total["cache"] / total["prompt"] * 100
                          if total["prompt"] else 0),
        "records": records,
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")
    print(f"TOTAL: {total['n']} items, ${total['cost']:.2f}, "
          f"cache {(total['cache']/max(total['prompt'],1))*100:.0f}%")


if __name__ == "__main__":
    main()
