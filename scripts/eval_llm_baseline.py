"""LLM baseline eval: give the LLM the full call history so far + source code,
ask it to simulate the next response, compare to ground truth.

Tests two memory strategies:
  'full_history' : feed EVERY prior (op,args,response) tuple
  'sliding'      : feed only the last K calls (K=20)

This is the MIRAGE-style baseline: LLM maintains state implicitly via prompt.
The TPS cost scales linearly with trajectory length, and we expect it to
degrade on long-horizon evals where cart state grows.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from invariants import check_response_invariants


def make_prompt(source_code, prior_calls, cur_op, cur_args):
    prior = "\n".join(
        f"CALL {i+1}: {c['op']}({json.dumps(c['args'])}) -> {json.dumps(c['response'])}"
        for i, c in enumerate(prior_calls)
    )
    return f"""DEPENDENCY SOURCE:
{source_code}

PRIOR CALL HISTORY (in order):
{prior if prior else '(no prior calls)'}

NEW REQUEST:
{cur_op}({json.dumps(cur_args)})

Simulate this dependency. Output ONLY the JSON response object this dependency would
produce given the prior call history. Be faithful to the source code's state
transitions. ONLY JSON, no prose, no code fences."""


SYSTEM = "You simulate a stateful microservice dependency exactly as its source code specifies. Output ONLY valid JSON. No prose."


def call_llm(model_id, prompt, litellm_mod, max_tokens=300):
    """Requires env var AWS_BEARER_TOKEN_BEDROCK + AWS_REGION for Bedrock models."""
    import os
    os.environ.setdefault("AWS_REGION", "us-east-1")
    kwargs = dict(
        model=model_id,
        messages=[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],
        max_tokens=max_tokens, timeout=60,
    )
    r = litellm_mod.completion(**kwargs)
    return r.choices[0].message.content or ""


def parse_json_loose(s: str):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lstrip().startswith("json"): s = s.lstrip()[4:]
    # grab first {...} block
    depth = 0; start = None; end = None
    for i, c in enumerate(s):
        if c == "{":
            if depth == 0: start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start is not None:
                end = i + 1; break
    if start is None or end is None:
        return None
    try:
        return json.loads(s[start:end])
    except Exception:
        return None


def eval_traj(traj, model_id, strategy, window, litellm_mod, concurrent_calls=1,
              step_budget=200):
    """Sequentially simulate trajectory. Can't parallelize within a traj because
    each call depends on prior history (though sliding window also sequential)."""
    source = traj["source_code"]
    steps = traj["steps"][:step_budget]
    out = []
    history = []
    for step in steps:
        op = step["op"]; args = step["args"]; gt = step["response"]
        if strategy == "sliding":
            prior = history[-window:]
        else:
            prior = history
        prompt = make_prompt(source, prior, op, args)
        t0 = time.time()
        try:
            resp_text = call_llm(model_id, prompt, litellm_mod)
            dt = time.time() - t0
            parsed = parse_json_loose(resp_text)
        except Exception as e:
            parsed = None; dt = time.time() - t0; resp_text = f"ERR:{e}"
        status_gt = "error" in gt if isinstance(gt, dict) else False
        if parsed is None:
            wf = False; sm = False; vm = False; viols = ["PARSE_FAIL"]
        else:
            wf = True
            status_pred = "error" in parsed if isinstance(parsed, dict) else False
            sm = (status_pred == status_gt)
            vm = parsed == gt
            viols = check_response_invariants(op, args, parsed)
        out.append({"step":step["step"], "op":op, "well_formed":wf,
                    "status_match":sm, "value_match":vm, "violations":viols,
                    "latency_s":round(dt,2),
                    "prompt_len":len(prompt),
                    "parsed":parsed, "ground_truth":gt,
                    "raw":resp_text[:150]})
        # Use GROUND TRUTH as history (we're measuring how well LLM simulates,
        # not how much errors compound — standard practice for sim-fidelity eval)
        history.append({"op":op, "args":args, "response":gt})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--model", default="bedrock/moonshotai.kimi-k2.5")
    ap.add_argument("--strategy", choices=["full_history","sliding"], default="sliding")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--n-trajs", type=int, default=6,
                    help="few trajs — each is sequential, so slow")
    ap.add_argument("--step-budget", type=int, default=200)
    ap.add_argument("--max-traj-concurrency", type=int, default=3,
                    help="run N trajs in parallel (each internally sequential)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import litellm
    litellm.drop_params = True

    trajs = []
    with open(args.eval_jsonl) as f:
        for i, line in enumerate(f):
            if i >= args.n_trajs: break
            t = json.loads(line)
            t["steps"] = t["steps"][:args.step_budget]
            trajs.append(t)

    results = [None] * len(trajs)
    with ThreadPoolExecutor(max_workers=args.max_traj_concurrency) as pool:
        futs = {pool.submit(eval_traj, t, args.model, args.strategy, args.window, litellm,
                            1, args.step_budget): i for i, t in enumerate(trajs)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
                wf = sum(1 for s in results[i] if s["well_formed"])
                vm = sum(1 for s in results[i] if s["value_match"])
                print(f"  traj {i}: {len(results[i])} steps, wf={wf}, vm={vm}", flush=True)
            except Exception as e:
                print(f"  traj {i} FAILED: {e}", flush=True)
                results[i] = []

    # Aggregate by step bucket
    from collections import defaultdict
    agg = defaultdict(lambda: {"n":0,"wf":0,"sm":0,"vm":0,"viol":0,"lat":0.0})
    for r in results:
        for s in r:
            b = s["step"] // 10
            a = agg[b]
            a["n"] += 1; a["wf"] += int(s["well_formed"])
            a["sm"] += int(s["status_match"]); a["vm"] += int(s["value_match"])
            a["viol"] += len(s["violations"])
            a["lat"] += s["latency_s"]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "strategy": args.strategy, "window": args.window,
        "n_trajs": len(trajs), "step_budget": args.step_budget,
        "aggregate_by_bucket": [{"bucket":b, **v} for b, v in sorted(agg.items())],
        "per_trajectory_summary": [
            {"traj_idx":i, "n_steps":len(r),
             "well_formed":sum(1 for s in r if s["well_formed"]),
             "status_match":sum(1 for s in r if s["status_match"]),
             "value_match":sum(1 for s in r if s["value_match"]),
             "total_violations":sum(len(s["violations"]) for s in r),
             "total_latency_s":round(sum(s["latency_s"] for s in r),1)}
            for i, r in enumerate(results)
        ],
    }, indent=2))
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
