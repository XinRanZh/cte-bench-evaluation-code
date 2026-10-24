"""Standalone runner for Phase C only, using the state-probe v2 grader.

Reads items from a counterfactuals JSONL (the authoritative catalog),
asks one LLM to produce the final hidden state per item, scores with
grade_state_probe_v2 (predeclared testbed paths + null-baseline
subtraction), streams every call to disk the moment it returns.

No Phase A, no Phase B. One call per item. Cheap to run on any model
we already ran Phase A+B for, so we can attach a v2 SP column to
headline results without re-running the whole eval.

Resume: every cached record in <calls-dir>/calls.jsonl carries `run_config`
(model, thinking, max_tokens, prompt history length, routing and inference
settings, catalog sha256) and
`catalog_row_sha256`. A rerun reuses cached records only if all of them were
produced under the same configuration; otherwise it stops, unless
--allow-mixed is given, in which case the output is labelled
mixed_run_configurations=true and lists every configuration. A record whose
call failed (provider error) is retried. The output holds one record per
selected catalog row (the newest; --max-items applies), and total_cost_usd
and cache_hit_pct are computed over exactly those records; `invocation`
reports this run's new calls.

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
    call_llm, make_state_prompt, request_settings, SYSTEM_STATE,
)
from grader import parse_json_loose, grade_state_probe, grade_state_probe_v2
from testbeds import TESTBEDS
import run_provenance as rp  # noqa: E402

MAX_HISTORY = 60
MIXED_MARKER_KEY = "_mixed_run_configurations"
STRUCTURAL_ERRORS = ("MISSING_GROUND_TRUTH_STATES", "NO_PROBE_SPEC_V2")


def failed_call(rec: dict) -> bool:
    """A record whose model call failed (provider/infrastructure error): retried on resume."""
    return bool(rec.get("infra_error")) or bool(rec.get("error") and rec.get("error") not in STRUCTURAL_ERRORS)


def state_probe_config(args) -> dict:
    return {"tool": "run_state_probe_v2", "model": args.model, "thinking": args.thinking,
            "max_tokens": args.max_tokens, "max_history": MAX_HISTORY,
            "request_settings": request_settings(args.model),
            "cf_jsonl_sha256": rp.file_sha256(args.cf_jsonl)}


def _attach_source(item: dict) -> None:
    if item.get("source_code"):
        return
    mod = TESTBEDS[item["domain"]]
    item["source_code"] = mod.SOURCE_CODE


def probe_item(item: dict, model_id: str, max_tokens: int,
               thinking_mode: str, per_call_logger=None,
               run_config: dict | None = None, row_id: str | None = None) -> dict:
    stamp = {rp.ROW_CONFIG_KEY: run_config, rp.ROW_ID_KEY: row_id} if run_config else {}
    _attach_source(item)
    if item.get("final_state_cf") is None or item.get("final_state_original") is None:
        return {"item_id": item.get("item_id"),
                "domain": item.get("domain"),
                "error": "MISSING_GROUND_TRUTH_STATES", **stamp}

    domain_mod = TESTBEDS.get(item["domain"])
    probe_spec = getattr(domain_mod, "PROBE_PATHS_V2", None)
    if not probe_spec:
        return {"item_id": item.get("item_id"),
                "domain": item.get("domain"),
                "error": "NO_PROBE_SPEC_V2", **stamp}

    prompt = make_state_prompt(item, max_history=MAX_HISTORY)
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
        # a provider/infrastructure error, not a model output: retried on resume
        **({"infra_error": call_err} if call_err else {}),
        "probe_v1": probe_v1,
        "probe_v2": probe_v2,
        **stamp,
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
    ap.add_argument("--allow-mixed", action="store_true",
                    help="Reuse cached records produced under another configuration (model, "
                         "thinking, max_tokens, catalog); the output is then labelled "
                         "mixed_run_configurations=true and lists every configuration.")
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

    run_config = state_probe_config(args)
    uids = [rp.catalog_row_id(it) for it in items]

    # Resume: a cached record is reused only if it was produced under this
    # configuration (or with --allow-mixed, which labels the output).
    cached, markers = [], []
    if calls_path.exists():
        with calls_path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if MIXED_MARKER_KEY in r:
                    markers.append(r)
                elif r.get("item_id"):
                    cached.append(r)
    foreign = [r for r in cached if r.get(rp.ROW_CONFIG_KEY) != run_config or not r.get(rp.ROW_ID_KEY)]
    problems = []
    if foreign:
        ex = foreign[0].get(rp.ROW_CONFIG_KEY)
        problems.append(f"{len(foreign)} of {len(cached)} cached records in {calls_path} were not produced "
                        f"under this run's configuration (e.g. {foreign[0].get('item_id')}: "
                        f"{'no recorded configuration' if not isinstance(ex, dict) else ex})")
    if markers:
        problems.append(f"{calls_path} is marked as mixing run configurations")
    if problems and not args.allow_mixed:
        raise SystemExit("error: " + "; ".join(problems) + ". Use another --calls-dir, or pass --allow-mixed.")
    mixed = bool(problems)
    # Records are matched to catalog rows by identity only; a record without one
    # (older runner) is never reused, since item_ids can repeat across variants.
    # Records whose call failed are not reused; a later record for the same row supersedes them.
    latest = {}
    for r in cached:
        if r.get(rp.ROW_ID_KEY):
            latest[r[rp.ROW_ID_KEY]] = r
    done_ids = {u for u, r in latest.items() if not failed_call(r)}
    if mixed:
        print("[resume] WARNING (--allow-mixed): " + "; ".join(problems))
    if done_ids:
        print(f"[resume] {len(done_ids & set(uids))} items already logged; skipping them")

    pending = [(it, u) for it, u in zip(items, uids) if u not in done_ids]
    print(f"evaluating {len(pending)}/{len(items)} items "
          f"with model={args.model} thinking={args.thinking}")

    calls_f = calls_path.open("a")
    if mixed:
        calls_f.write(json.dumps({MIXED_MARKER_KEY: {"ts": time.time(), "run_config": run_config,
                                                     "problems": problems}}) + "\n")
        calls_f.flush()
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
                            args.thinking, logger, run_config, u): it["item_id"]
                for it, u in pending}
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

    # Final aggregate dump: one record per selected catalog row (this run's items,
    # after --max-items), the newest logged one; every aggregate uses exactly these.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    selected = set(uids)
    latest = {}
    with calls_path.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if MIXED_MARKER_KEY not in rec and rec.get(rp.ROW_ID_KEY) in selected:
                latest[rec[rp.ROW_ID_KEY]] = rec
    records = [latest[u] for u in uids if u in latest]
    rec_prompt = sum((r.get("usage") or {}).get("prompt_tokens", 0) or 0 for r in records)
    rec_cache = sum((r.get("usage") or {}).get("cached_tokens", 0) or 0 for r in records)
    configs: dict = {}
    for rec in records:
        k = json.dumps(rec.get(rp.ROW_CONFIG_KEY), sort_keys=True)
        configs[k] = configs.get(k, 0) + 1
    out_path.write_text(json.dumps({
        "model": args.model,
        "thinking": args.thinking,
        "run_config": run_config,
        "mixed_run_configurations": mixed,
        **({"provenance_problems": problems,
            "row_run_configurations": [{"run_config": json.loads(k), "n_records": n} for k, n in configs.items()]}
           if mixed else {}),
        "n_records": len(records),
        "n_failed_calls": sum(failed_call(r) for r in records),
        # cost and cache rate of the emitted records (cached and new)
        "total_cost_usd": sum(r.get("cost_usd", 0) or 0 for r in records),
        "cache_hit_pct": (rec_cache / rec_prompt * 100) if rec_prompt else 0,
        # this invocation's new calls only
        "invocation": {"n_calls": total["n"], "cost_usd": total["cost"],
                       "cache_hit_pct": (total["cache"] / total["prompt"] * 100) if total["prompt"] else 0},
        "records": records,
    }, indent=2, default=str))
    if mixed:
        print(f"WARNING: MIXED RUN PROVENANCE: records come from {len(configs)} configuration(s); "
              f"mixed_run_configurations=true")
    print(f"\nwrote {out_path}")
    print(f"THIS RUN: {total['n']} calls, ${total['cost']:.2f}, "
          f"cache {(total['cache']/max(total['prompt'],1))*100:.0f}%")
    print(f"OUTPUT: {len(records)} records, ${sum(r.get('cost_usd', 0) or 0 for r in records):.2f}, "
          f"cache {(rec_cache/max(rec_prompt,1))*100:.0f}%")


if __name__ == "__main__":
    main()
