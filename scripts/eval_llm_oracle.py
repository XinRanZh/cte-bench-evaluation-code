"""LLM baseline with ORACLE state-summary prefix — Pivot 1 feasibility check.

Question: if the LLM is handed a perfect summary of the hidden program state at
each call (instead of reconstructing from history), can it maintain high value
fidelity at 200 steps?

- If YES: a LEARNED compressor of history → prefix can achieve similar lift.
           Pivot 1 (LLM + learned state compressor) is viable.
- If NO:   LLM has a deeper limit; compressor won't save it. Switch to Pivot 2/3.

Three conditions:
  strategy='sliding_only'    — LLM + last 20 calls (v2 default; already have data)
  strategy='oracle_only'     — LLM + NL summary of true state, no call history
  strategy='oracle_sliding'  — LLM + oracle state + last 20 calls (upper bound)

Reads the same eval trajectories as v2; ground-truth state_after is in each step.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parent))
from invariants import check_response_invariants


def summarize_state_natural(state: dict, domain: str) -> str:
    """Render the hidden state_after into a compact natural-language / structured
    summary that the LLM can attend to. Output length ~50-200 tokens.

    For Pivot 1 the equivalent learned compressor would produce ~50 tokens of
    "state prefix" that plays the same role.
    """
    if domain == "bank":
        accts = state.get("accounts", {})
        closed = state.get("closed", [])
        td = state.get("total_deposits", 0.0)
        tw = state.get("total_withdrawals", 0.0)
        lines = ["STATE:"]
        if accts:
            lines.append("  accounts (active):")
            for aid, bal in sorted(accts.items()):
                lines.append(f"    {aid}: balance={bal:.2f}")
        else:
            lines.append("  accounts (active): none")
        if closed:
            lines.append(f"  accounts (closed): {', '.join(sorted(closed))}")
        lines.append(f"  totals: deposits={td:.2f}, withdrawals={tw:.2f}")
        return "\n".join(lines)

    if domain == "cart":
        carts = state.get("carts", {})
        lines = ["STATE:"]
        if carts:
            for uid, items in sorted(carts.items()):
                if items:
                    items_s = ", ".join(f"{it['product_id']} x{it['quantity']}"
                                        for it in items)
                    lines.append(f"  {uid}: {items_s}")
                else:
                    lines.append(f"  {uid}: (empty)")
        else:
            lines.append("  carts: none")
        return "\n".join(lines)

    if domain == "auth":
        toks = state.get("tokens", {})
        revoked = set(state.get("revoked", []))
        clock = state.get("clock", 0)
        lines = [f"STATE: clock={clock}"]
        if toks:
            lines.append("  tokens:")
            for tid, t in sorted(toks.items()):
                alive = "ALIVE" if t["alive"] else "DEAD"
                rev = "(revoked)" if tid in revoked else ""
                lines.append(f"    {tid}: user={t['user_id']} "
                             f"ttl={t['ttl']} issued_at={t['issued_at']} {alive} {rev}")
        else:
            lines.append("  tokens: none")
        return "\n".join(lines)

    return "STATE: (unknown domain)"


def make_prompt_oracle(source_code, state_text, prior_calls, cur_op, cur_args,
                       use_history=False):
    prior = ""
    if use_history and prior_calls:
        prior = "\nRECENT CALLS:\n" + "\n".join(
            f"  {c['op']}({json.dumps(c['args'])}) -> {json.dumps(c['response'])}"
            for c in prior_calls
        ) + "\n"
    return f"""DEPENDENCY SOURCE:
{source_code}

{state_text}
{prior}
NEW REQUEST: {cur_op}({json.dumps(cur_args)})

Produce the JSON response this dependency would return given the STATE above.
Be faithful to the state and source. ONLY JSON."""


SYSTEM = ("You simulate a stateful microservice dependency exactly as its source "
          "code specifies. Output ONLY valid JSON. No prose.")


def call_llm(model_id, prompt, litellm_mod, max_tokens=300):
    os.environ.setdefault("AWS_REGION", "us-east-1")
    kwargs = dict(model=model_id,
                  messages=[{"role":"system","content":SYSTEM},
                            {"role":"user","content":prompt}],
                  max_tokens=max_tokens, timeout=60)
    r = litellm_mod.completion(**kwargs)
    return r.choices[0].message.content or ""


def parse_json_loose(s: str):
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```")[1]
        if s.lstrip().startswith("json"): s = s.lstrip()[4:]
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


def eval_traj(traj, model_id, strategy, window, litellm_mod, step_budget=200):
    source = traj["source_code"]
    domain = traj["domain"]
    steps = traj["steps"][:step_budget]
    out = []
    history = []
    for step in steps:
        op = step["op"]; args = step["args"]; gt = step["response"]
        # Oracle uses state *before* this call; state_after from the PREVIOUS step
        if strategy in ("oracle_only", "oracle_sliding"):
            if len(out) == 0:
                # very first call — state before anything is done
                prev_state = {"accounts": {}, "closed": [], "total_deposits": 0.0,
                              "total_withdrawals": 0.0, "carts": {},
                              "tokens": {}, "revoked": [], "clock": 0, "counter": 0}
            else:
                prev_state = steps[len(out) - 1]["state_after"]
            state_text = summarize_state_natural(prev_state, domain)
        else:
            state_text = "(state not provided)"

        use_hist = strategy in ("sliding_only", "oracle_sliding")
        prior = history[-window:] if use_hist else []
        prompt = make_prompt_oracle(source, state_text, prior, op, args, use_history=use_hist)

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
        history.append({"op":op, "args":args, "response":gt})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--model", default="bedrock/moonshotai.kimi-k2.5")
    ap.add_argument("--strategy", choices=["sliding_only","oracle_only","oracle_sliding"],
                    required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--n-trajs", type=int, default=3)
    ap.add_argument("--step-budget", type=int, default=200)
    ap.add_argument("--max-traj-concurrency", type=int, default=3)
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
        futs = {pool.submit(eval_traj, t, args.model, args.strategy, args.window,
                            litellm, args.step_budget): i for i, t in enumerate(trajs)}
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
        "aggregate_by_bucket":[{"bucket":b,**v} for b,v in sorted(agg.items())],
        "per_trajectory_summary":[
            {"traj_idx":i,"n_steps":len(r),
             "well_formed":sum(1 for s in r if s["well_formed"]),
             "status_match":sum(1 for s in r if s["status_match"]),
             "value_match":sum(1 for s in r if s["value_match"]),
             "total_violations":sum(len(s["violations"]) for s in r),
             "total_latency_s":round(sum(s["latency_s"] for s in r),1)}
            for i,r in enumerate(results)
        ],
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
