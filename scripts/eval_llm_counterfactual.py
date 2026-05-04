"""LLM counterfactual eval — feeds CTE-Bench items through a model.

For each counterfactual item this script performs up to three distinct LLM
calls and grades all four headline metrics:

  Phase A — counterfactual trace prediction (one call per suffix step):
    The model sees {original source, intervention description, prefix trace}
    and predicts the response to each suffix step under the counterfactual.
    Scored by grade_value_match + grade_status_match.

    In TF-1step modes (`full_history` and `sliding`), we feed the
    GROUND-TRUTH counterfactual response back into the conversation as
    'prior history', not the model's previous prediction. This isolates the
    fidelity of each single-step simulation from compounding-error dynamics.
    In `free_rollout`, prior suffix history is instead populated with the
    model's own previous predictions.

  Phase B — explanation localization (one call, code/adversarial items only):
    The model is told that one line of source was changed and asked to name
    the line number. Scored by grade_explanation_localization.

  Phase C — terminal state probe (one call, sampled subset):
    The model is asked to output the final state dict after applying the
    entire suffix. Scored by grade_state_probe on a few paths.

Memory strategy (configurable):
  - `full_history`   : every prior suffix step goes into the prompt
  - `sliding`        : only the last K steps go into the prompt
  - `prefix_only`    : only the prefix goes in; suffix prior steps are omitted.
                       Forces pure latent-state abduction.
  - `free_rollout`   : like sliding K, but prior suffix responses are the
                       model's own predictions instead of oracle responses.

Concurrency: per-item sequential (the suffix calls share conversational
state), but many items can run in parallel via ThreadPoolExecutor.
"""
from __future__ import annotations
import argparse, copy, json, os, random, sys, threading, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from grader import (parse_json_loose, grade_value_match, grade_status_match,
                    grade_explanation_localization, grade_state_probe,
                    grade_state_probe_v2, extract_line_number)
from core_v1 import normalize_line_number_convention, normalize_source_text
from testbeds import TESTBEDS


SYSTEM = ("You simulate a stateful service exactly as its source code specifies. "
          "When asked for a response, output ONLY valid JSON. No prose, "
          "no code fences.")

SYSTEM_EXPLAIN = ("You read Python source and identify the SINGLE line number "
                  "that was changed between two versions of a program. Respond "
                  "with ONLY a JSON object {\"line_number\": N}.")

SYSTEM_STATE = ("You output the final hidden state of a stateful service after "
                "a sequence of operations. Respond with ONLY a JSON object "
                "matching the service's state schema.")


# ---------------------------------------------------------------------------
# Prompt builders

def _fmt_history(calls: list, max_items: int | None = None) -> str:
    if max_items is not None:
        calls = calls[-max_items:]
    if not calls:
        return "(none)"
    return "\n".join(
        f"CALL {i+1}: {c['op']}({json.dumps(c['args'])}) -> {json.dumps(c['response'])}"
        for i, c in enumerate(calls)
    )


def _fmt_intervention(spec: dict) -> str:
    kind = spec.get("kind")
    if kind in ("threshold_change", "branch_inversion", "quota_change", "adversarial"):
        return (f"INTERVENTION (applied at step {spec.get('line_number','?')}):\n"
                f"  kind: {kind}\n"
                f"  at source line {spec['line_number']}, the token "
                f"`{spec['original_span']}` has been replaced by "
                f"`{spec['modified_span']}`.\n"
                f"  description: {spec.get('description','')}")
    if kind in ("state_injection", "clock_jump"):
        return (f"INTERVENTION (applied at step k in the trace):\n"
                f"  kind: {kind}\n"
                f"  at the moment AFTER prefix step {spec.get('step_k','?')} "
                f"completed, the hidden state path {spec.get('path')} was "
                f"overwritten from {spec.get('from_value')!r} to "
                f"{spec.get('to_value')!r}.\n"
                f"  The SOURCE CODE is unchanged. Only the hidden state was "
                f"mutated between step k-1 and step k.")
    return f"INTERVENTION: {spec}"


def make_counterfactual_prompt(item: dict, suffix_history: list,
                               cur_op: str, cur_args: dict,
                               strategy: str, window: int) -> str:
    """Assemble the prompt for one suffix step under a counterfactual."""
    # Pick source: code/adversarial interventions expose the PATCHED source;
    # state interventions use the original source.
    kind = item["intervention"].get("kind")
    if kind in ("state_injection", "clock_jump"):
        # Use the original source (we don't have it here directly; it's the
        # source saved on the item). Note: the generator saves `patched_source`
        # only for code interventions; state items don't save source. We rely
        # on caller to inject item["source_code"]. For safety, fall back.
        source_code = item.get("source_code")
    else:
        source_code = item.get("patched_source") or item.get("source_code")

    prefix = item["trace_prefix"]
    # Assemble prior calls by strategy:
    if strategy == "prefix_only":
        prior_all = prefix
    else:
        prior_all = prefix + suffix_history

    if strategy in ("sliding", "free_rollout"):
        prior_str = _fmt_history(prior_all, max_items=window)
    else:
        prior_str = _fmt_history(prior_all)

    intervention_block = _fmt_intervention({**item["intervention"],
                                            "step_k": item["prefix_len"]})

    return f"""DEPENDENCY SOURCE (possibly patched):
{source_code}

{intervention_block}

PRIOR CALL HISTORY (under the counterfactual, in order):
{prior_str}

NEW REQUEST:
{cur_op}({json.dumps(cur_args)})

Simulate this dependency under the counterfactual. Output ONLY the JSON
response object this dependency would produce. No prose, no code fences."""


def make_explain_prompt(item: dict) -> str:
    # We need ORIGINAL source + PATCHED source side by side for the model.
    item = normalize_line_number_convention(item)
    original_source = item.get("source_code")
    patched_source  = item.get("patched_source")
    if patched_source is None or original_source is None:
        return None
    return f"""Two versions of a Python service were shown to a model. One
single line of the source was changed between versions. Name the changed
line number from the line-numbered PATCHED SOURCE.

ORIGINAL SOURCE:
{_format_numbered_source(original_source)}

PATCHED SOURCE:
{_format_numbered_source(patched_source)}

Respond with ONLY a JSON object: {{"line_number": N}}"""


def _format_numbered_source(source: str) -> str:
    source = normalize_source_text(source)
    return "\n".join(f"{i:>3}: {line}" for i, line in enumerate(source.splitlines(), 1))


def make_state_prompt(item: dict, max_history: int | None = None) -> str:
    kind = item["intervention"].get("kind")
    source_code = (item.get("patched_source") if kind not in
                   ("state_injection", "clock_jump") else item.get("source_code"))
    suffix = item["trace_counterfactual"]
    prefix = item["trace_prefix"]
    all_calls = prefix + suffix
    history_str = _fmt_history(all_calls, max_items=max_history)
    intervention_block = _fmt_intervention({**item["intervention"],
                                            "step_k": item["prefix_len"]})
    return f"""DEPENDENCY SOURCE:
{source_code}

{intervention_block}

FULL CALL HISTORY (under the counterfactual, in order):
{history_str}

Output the service's complete final hidden state AS A JSON OBJECT, matching
the schema that state_dict() would return. No prose, no code fences."""


# ---------------------------------------------------------------------------
# LLM caller (mirrors eval_llm_baseline.call_llm)

# ---------------------------------------------------------------------------
# Pricing table (USD per 1M tokens, as of 2026-04).
# For models not listed we report token counts only; cost stays 0.

PRICING_USD_PER_1M = {
    # (input_miss, output, input_cache_hit)  — cache_hit optional (default 10% of miss)
    #
    # Anthropic / Bedrock (cache_hit = 10% per Anthropic standard prompt caching)
    "bedrock/us.anthropic.claude-opus-4-7":                   (15.00, 75.00, 1.50),
    "bedrock/us.anthropic.claude-opus-4-5-20251101-v1:0":     (15.00, 75.00, 1.50),
    "bedrock/us.anthropic.claude-sonnet-4-6":                 ( 3.00, 15.00, 0.30),
    "bedrock/global.anthropic.claude-sonnet-4-6":             ( 3.00, 15.00, 0.30),
    "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0":   ( 3.00, 15.00, 0.30),
    "bedrock/anthropic.claude-haiku-4-5-20251001-v1:0":       ( 1.00,  5.00, 0.10),
    # Moonshot Kimi via Bedrock (Bedrock cache applies at 10%)
    "bedrock/moonshotai.kimi-k2.5":                           ( 0.60,  2.50, 0.06),
    # DeepSeek V4 pricing, fetched 2026-05-01.
    # https://api-docs.deepseek.com/quick_start/pricing
    # Compatibility names: deepseek-chat = v4-flash non-thinking;
    # deepseek-reasoner = v4-flash thinking until 2026-07-24.
    "deepseek/deepseek-v4-pro":                               (1.74, 3.48, 0.145),
    "deepseek/deepseek-v4-flash":                             (0.14, 0.28, 0.028),
    "deepseek/deepseek-chat":                                 (0.14, 0.28, 0.028),
    "deepseek/deepseek-reasoner":                             (0.14, 0.28, 0.028),
    # OpenRouter model page prices fetched 2026-05-01.
    # https://openrouter.ai/qwen/qwen3.6-35b-a3b endpoint tag parasail/fp8
    "openrouter/qwen/qwen3.6-35b-a3b":                          (0.20, 1.00, 0.05),
    # https://openrouter.ai/moonshotai/kimi-k2.5
    "openrouter/moonshotai/kimi-k2.5":                            (0.44, 2.00, 0.44),
    # https://openrouter.ai/google/gemma-4-31b-it
    "openrouter/google/gemma-4-31b-it":                         (0.130, 0.38, 0.130),
    # OpenAI (rough 2026-04 tier estimates; verify before billing)
    "openai/gpt-5.5":                                         (1.25, 10.00, 0.125),
    "openai/gpt-5.4":                                         (1.25, 10.00, 0.125),
    "openai/gpt-4.1":                                         (2.00,  8.00, 0.50),
    "openai/gpt-4o-mini":                                     (0.15,  0.60, 0.075),
    # Google Gemini
    "gemini/gemini-3.1-pro":                                  (1.25, 10.00, 0.3125),
}


def price_call(model_id: str, usage: dict) -> float:
    """Return $ cost of a single call, or 0 if model not in PRICING_USD_PER_1M.

    Per-provider billing rules (based on provider docs, 2026-04-30):
      - DeepSeek:
          * `prompt_cache_hit_tokens` (aliased to `cached_tokens` here) is billed
            at the cache-hit rate (10% of miss price per DS 2026-04-26 policy).
          * `completion_tokens` INCLUDES `reasoning_tokens`; do NOT double-count.
          * v4-pro is at a 75% discount through 2026-05-31 — the rates in
            PRICING_USD_PER_1M already reflect that.
      - Anthropic / Bedrock: `completion_tokens` is the billable output; if
        the SDK returns `reasoning_tokens` SEPARATELY (some thinking models do)
        we add it; otherwise it's already included.
      - OpenRouter: `completion_tokens` already includes reasoning tokens in
        the reported output total; `reasoning_tokens` is an audit breakdown.
      - OpenAI / Gemini: simple in+out. Cache unsupported here.
    `PRICING_USD_PER_1M` entries are (in_miss, out) in USD/1M. A third entry
    may be given for the cache-hit rate; if absent, we default to 10% of miss.
    """
    if model_id not in PRICING_USD_PER_1M:
        return 0.0
    rates = PRICING_USD_PER_1M[model_id]
    in_rate = rates[0]
    out_rate = rates[1]
    cache_rate = rates[2] if len(rates) >= 3 else in_rate * 0.10
    prompt = usage.get("prompt_tokens", 0) or 0
    cached = usage.get("cached_tokens", 0) or 0
    completion = usage.get("completion_tokens", 0) or 0
    reasoning = usage.get("reasoning_tokens", 0) or 0

    if model_id.startswith("deepseek/") or model_id.startswith("openrouter/"):
        # completion_tokens already includes reasoning_tokens for these APIs.
        billable_out = completion
    else:
        # Anthropic/OpenAI/Gemini: add reasoning if provided separately
        billable_out = completion + reasoning

    billable_miss = max(0, prompt - cached)
    return (billable_miss / 1e6 * in_rate
            + cached / 1e6 * cache_rate
            + billable_out / 1e6 * out_rate)


def _extract_usage(resp) -> dict:
    """Normalize litellm Usage to a plain dict, capturing the fields we bill on."""
    u = getattr(resp, "usage", None)
    if u is None:
        return {}
    out = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }
    # thinking models
    ctd = getattr(u, "completion_tokens_details", None)
    if ctd is not None:
        out["reasoning_tokens"] = getattr(ctd, "reasoning_tokens", 0) or 0
    # DeepSeek cache hits
    cached = getattr(u, "prompt_cache_hit_tokens", None)
    if cached is None:
        ptd = getattr(u, "prompt_tokens_details", None)
        if ptd is not None:
            cached = getattr(ptd, "cached_tokens", None)
    out["cached_tokens"] = cached or 0
    return out


def _retryable_llm_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(tok in msg for tok in (
        "429", "rate limit", "rate_limit", "temporarily", "timeout",
        "timed out", "connection", "overloaded", "502", "503", "504",
    ))


def _provider_kwargs(model_id: str, max_tokens: int, thinking_mode: str,
                     reasoning_effort: str = "high"):
    """Return provider-specific kwargs for the API call.

    `thinking_mode` is one of {"off", "on"}:
      - "off": disable thinking. For DS Pro pass thinking={"type":"disabled"};
        for Kimi K2.5 pass extra_body thinking={"type":"disabled"};
        for Anthropic pass nothing.
      - "on": enable thinking. For DS Pro pass the requested reasoning_effort;
        for Kimi K2.5 pass extra_body thinking={"type":"enabled"}; for Anthropic
        pass thinking={type:enabled, budget_tokens:...} and raise max_tokens
        above the budget.
    DS Flash (deepseek-v4-flash) is always off.
    """
    kw = dict(max_tokens=max_tokens)
    if model_id.startswith("deepseek/"):
        if "deepseek-chat" in model_id:
            # legacy non-thinking alias — does not accept thinking kwarg
            pass
        elif "flash" in model_id:
            # v4-flash ignores disabled and keeps reasoning; callers who want a
            # strict non-thinking baseline should use `deepseek-chat` instead.
            kw["thinking"] = {"type": "disabled"}
        elif thinking_mode == "on":
            kw["thinking"] = {"type": "enabled",
                              "reasoning_effort": reasoning_effort}
        else:
            kw["thinking"] = {"type": "disabled"}
    elif "anthropic" in model_id and thinking_mode == "on":
        budget = min(2048, max(512, max_tokens // 2))
        kw["max_tokens"] = max(kw["max_tokens"], budget + max_tokens)
        kw["thinking"] = {"type": "enabled", "budget_tokens": budget}
    elif model_id.startswith("openrouter/"):
        extra_body = {}
        provider = os.environ.get("OPENROUTER_PROVIDER")
        if provider:
            extra_body["provider"] = {
                "only": [p.strip() for p in provider.split(",") if p.strip()],
                "allow_fallbacks": False,
            }
        quantizations = os.environ.get("OPENROUTER_QUANTIZATIONS")
        if quantizations:
            extra_body.setdefault("provider", {})
            extra_body["provider"]["quantizations"] = [
                q.strip() for q in quantizations.split(",") if q.strip()
            ]
        if thinking_mode == "on":
            reasoning = {"enabled": True, "exclude": True}
            max_reasoning = os.environ.get("OPENROUTER_REASONING_MAX_TOKENS")
            if max_reasoning:
                reasoning["max_tokens"] = int(max_reasoning)
            else:
                reasoning["effort"] = os.environ.get(
                    "OPENROUTER_REASONING_EFFORT", reasoning_effort)
            extra_body["reasoning"] = reasoning
        else:
            extra_body["reasoning"] = {"effort": "none", "exclude": True}
        kw["extra_body"] = extra_body
    elif "moonshotai.kimi-k2.5" in model_id:
        # Moonshot Kimi K2.5 exposes a binary thinking switch. The native API
        # defaults to enabled, so pass the flag explicitly for clean controls.
        kw["extra_body"] = {
            "thinking": {
                "type": "enabled" if thinking_mode == "on" else "disabled"
            }
        }
    return kw


_DS_CLIENT = None


def _get_ds_client():
    """Direct DeepSeek OpenAI-compatible client. litellm's wrapping of the
    DS chat endpoint has hung on long-max_tokens+thinking workloads for us;
    the DS docs recommend the OpenAI SDK with base_url=https://api.deepseek.com."""
    global _DS_CLIENT
    if _DS_CLIENT is None:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not set")
        _DS_CLIENT = OpenAI(api_key=api_key,
                            base_url="https://api.deepseek.com")
    return _DS_CLIENT


def _ds_model_name(model_id: str) -> str:
    """Map our 'deepseek/deepseek-v4-pro' slug to DS's actual model name."""
    tail = model_id.split("/", 1)[-1]
    # DS accepts deepseek-chat, deepseek-v4-pro, deepseek-v4-flash, etc.
    return tail


def _call_deepseek_direct(model_id: str, system: str, prompt: str,
                          max_tokens: int, timeout: int, thinking_mode: str,
                          reasoning_effort: str = "high"):
    """Direct DS call via OpenAI SDK. Returns (text, usage_dict, cost_usd)."""
    client = _get_ds_client()
    model = _ds_model_name(model_id)
    body = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": prompt}],
        max_tokens=max_tokens,
        timeout=timeout,
    )
    # DS thinking control per official docs:
    #   v4-pro: thinking={type:enabled|disabled, reasoning_effort:low|medium|high}
    #   deepseek-chat: legacy non-thinking alias — do not pass thinking
    if "deepseek-chat" in model:
        pass
    elif thinking_mode == "on":
        body["extra_body"] = {"thinking":
                              {"type": "enabled",
                               "reasoning_effort": reasoning_effort}}
    else:
        body["extra_body"] = {"thinking": {"type": "disabled"}}

    resp = client.chat.completions.create(**body)
    text = (resp.choices[0].message.content or "")

    # usage: DS returns prompt_cache_hit_tokens / prompt_cache_miss_tokens at
    # the top level of usage, and reasoning_tokens under completion_tokens_details.
    u = resp.usage
    usage = {
        "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
        "total_tokens": getattr(u, "total_tokens", 0) or 0,
    }
    ctd = getattr(u, "completion_tokens_details", None)
    if ctd is not None:
        r_tok = getattr(ctd, "reasoning_tokens", None)
        # Sometimes the SDK returns dict-like
        if r_tok is None and isinstance(ctd, dict):
            r_tok = ctd.get("reasoning_tokens")
        usage["reasoning_tokens"] = r_tok or 0
    cache_hit = getattr(u, "prompt_cache_hit_tokens", None)
    if cache_hit is None:
        ptd = getattr(u, "prompt_tokens_details", None)
        if ptd is not None:
            cache_hit = getattr(ptd, "cached_tokens", None)
    usage["cached_tokens"] = cache_hit or 0

    cost = price_call(model_id, usage)
    return text, usage, cost


def call_llm(model_id: str, system: str, prompt: str, litellm_mod,
             max_tokens: int = 4000, timeout: int = 1200,
             thinking_mode: str = "off",
             reasoning_effort: str = "high") -> tuple:
    """Return (content_text, usage_dict, cost_usd).

    thinking_mode: "on" enables reasoning/thinking where supported (DS Pro,
    Anthropic 4.x). "off" disables (default, required for DS Flash to work
    fairly, and for apples-to-apples comparison across non-thinking runs).

    DeepSeek is routed via the OpenAI SDK directly (not litellm) because
    litellm has hung on long-context + high-thinking DS Pro calls in our
    workload. Other providers still go through litellm.
    """
    if model_id.startswith("deepseek/"):
        return _call_deepseek_direct(model_id, system, prompt,
                                     max_tokens, timeout, thinking_mode,
                                     reasoning_effort)
    os.environ.setdefault("AWS_REGION", "us-east-1")
    if model_id.startswith("openrouter/qwen/") and thinking_mode == "off":
        # Qwen hybrid thinking models can ignore generic provider-side
        # reasoning=none on some routes. The documented chat control token is
        # the most reliable way to keep a non-thinking run auditable.
        prompt = f"{prompt}\n\n/no_think"
    kwargs = _provider_kwargs(model_id, max_tokens, thinking_mode,
                              reasoning_effort)
    retries = int(os.environ.get("CTE_LLM_RETRIES", "4"))
    for attempt in range(retries + 1):
        try:
            r = litellm_mod.completion(
                model=model_id,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": prompt}],
                timeout=timeout,
                **kwargs,
            )
            break
        except Exception as e:
            if attempt >= retries or not _retryable_llm_error(e):
                raise
            delay = min(60.0, 2 ** attempt + random.random())
            if os.environ.get("CTE_TRACE"):
                print(f"      retryable LLM error on attempt {attempt + 1}: {e}; "
                      f"sleeping {delay:.1f}s", flush=True)
            time.sleep(delay)
    text = r.choices[0].message.content or ""
    usage = _extract_usage(r)
    cost = price_call(model_id, usage)
    return text, usage, cost


# ---------------------------------------------------------------------------
# Attach original source to each item (the generator saved patched_source but
# not original; we re-attach from the testbed module so the eval is
# self-contained).

def _attach_original_source(item: dict) -> None:
    if "source_code" in item and item["source_code"]:
        item["source_code"] = normalize_source_text(item["source_code"])
        return
    from testbeds import TESTBEDS
    mod = TESTBEDS[item["domain"]]
    item["source_code"] = normalize_source_text(mod.SOURCE_CODE)


# ---------------------------------------------------------------------------
# Per-item eval

def eval_item(item: dict, model_id: str, strategy: str, window: int,
              litellm_mod, suffix_budget: int = 200,
              do_explain: bool = True, do_state: bool = True,
              thinking_mode: str = "off",
              max_tokens: int = 4000,
              reasoning_effort: str = "high",
              per_call_logger=None) -> dict:
    """Evaluate one item across Phase A (suffix trace) + B (localization) +
    C (state probe). If `per_call_logger` is provided it is called with a
    dict after every individual LLM call, so the caller can stream partial
    progress to disk without waiting for the whole item to finish.
    """
    _attach_original_source(item)
    item = normalize_line_number_convention(item)
    suffix_cf = item["trace_counterfactual"][:suffix_budget]

    phase_a = []
    suffix_history: list = []
    # Per-item billing accumulator.
    bill = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0,
            "cached_tokens": 0, "cost_usd": 0.0, "n_calls": 0}
    def _bill(usage: dict, cost: float):
        bill["n_calls"] += 1
        bill["cost_usd"] += cost
        for k in ("prompt_tokens", "completion_tokens",
                  "reasoning_tokens", "cached_tokens"):
            bill[k] += usage.get(k, 0) or 0

    def _emit_call(phase, suffix_idx, op, dt, usage, cost,
                   vm=None, sm=None, well_formed=None,
                   predicted=None, ground_truth=None,
                   extracted=None, error=None):
        if per_call_logger is None:
            return
        rec = {
            "ts": time.time(),
            "item_id": item.get("item_id"),
            "domain": item.get("domain"),
            "intervention_kind": item["intervention"].get("kind")
                                 if isinstance(item.get("intervention"), dict) else None,
            "phase": phase,
            "suffix_idx": suffix_idx,
            "op": op,
            "latency_s": round(dt, 2),
            "usage": usage,
            "cost_usd": cost,
            "value_match": vm,
            "status_match": sm,
            "well_formed": well_formed,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "extracted": extracted,
            "error": error,
        }
        try:
            per_call_logger(rec)
        except Exception:
            pass

    for i, step in enumerate(suffix_cf):
        op = step["op"]; args = step["args"]; gt = step["response"]
        prompt = make_counterfactual_prompt(item, suffix_history, op, args,
                                            strategy=strategy, window=window)
        if os.environ.get("CTE_TRACE"):
            print(f"    [{item.get('item_id','?')[:8]}] suffix {i} op={op} prompt_len={len(prompt)} ... calling",
                  flush=True)
        t0 = time.time()
        usage, cost, call_err = {}, 0.0, None
        try:
            raw, usage, cost = call_llm(model_id, SYSTEM, prompt, litellm_mod,
                                        max_tokens=max_tokens,
                                        thinking_mode=thinking_mode,
                                        reasoning_effort=reasoning_effort)
            dt = time.time() - t0
            parsed = parse_json_loose(raw)
            _bill(usage, cost)
            if os.environ.get("CTE_TRACE"):
                print(f"    [{item.get('item_id','?')[:8]}] suffix {i} dt={dt:.1f}s "
                      f"in={usage.get('prompt_tokens')} cache={usage.get('cached_tokens')} "
                      f"out={usage.get('completion_tokens')} reasoning={usage.get('reasoning_tokens')}",
                      flush=True)
        except Exception as e:
            raw = f"ERR:{e}"; parsed = None; dt = time.time() - t0
            call_err = str(e)[:200]
            if os.environ.get("CTE_TRACE"):
                print(f"    [{item.get('item_id','?')[:8]}] suffix {i} FAILED dt={dt:.1f}s err={e}",
                      flush=True)
        vm = grade_value_match(parsed, gt)
        sm = grade_status_match(parsed, gt)
        phase_a.append({
            "suffix_idx": i,
            "op": op, "args": args,
            "predicted": parsed,
            "ground_truth": gt,
            "value_match": vm["match"],
            "status_match": sm["match"],
            "well_formed": vm.get("well_formed", False),
            "latency_s": round(dt, 2),
            "prompt_len": len(prompt),
            "raw_excerpt": raw[:160],
        })
        # Stream the call record to disk immediately.
        _emit_call("A", i, op, dt, usage, cost,
                   vm=vm["match"], sm=sm["match"],
                   well_formed=vm.get("well_formed", False),
                   predicted=parsed, ground_truth=gt, error=call_err)
        # TF-1step modes use oracle suffix history; free-rollout uses the
        # model's own previous response, so errors can compound.
        if strategy == "free_rollout":
            hist_response = parsed if parsed is not None else {
                "error": {"code": "MODEL_INVALID_JSON"}
            }
        else:
            hist_response = gt
        suffix_history.append({"op": op, "args": args, "response": hist_response})

    # --- Phase B: explanation localization (code/adversarial only)
    phase_b = None
    kind = item["intervention"].get("kind")
    if do_explain and kind in ("threshold_change", "branch_inversion",
                               "quota_change", "adversarial"):
        prompt_e = make_explain_prompt(item)
        if prompt_e:
            t0 = time.time()
            usage, cost, call_err = {}, 0.0, None
            try:
                raw, usage, cost = call_llm(model_id, SYSTEM_EXPLAIN, prompt_e,
                                            litellm_mod, max_tokens=2000,
                                            thinking_mode=thinking_mode,
                                            reasoning_effort=reasoning_effort)
                dt = time.time() - t0
                _bill(usage, cost)
            except Exception as e:
                raw = f"ERR:{e}"; dt = time.time() - t0; call_err = str(e)[:200]
            truth = item["explanation_truth"]
            pred_line = extract_line_number(raw)
            loc = grade_explanation_localization(pred_line, truth, tolerance=1)
            phase_b = {"raw": raw[:160], "latency_s": round(dt, 2),
                       "extracted_line": pred_line, **loc}
            _emit_call("B", None, "explain", dt, usage, cost,
                       extracted=pred_line, error=call_err)

    # --- Phase C: terminal state probe (v1 for compatibility + v2 with
    # predeclared paths + null-baseline subtraction)
    phase_c = None
    if do_state and item.get("final_state_cf") is not None:
        prompt_s = make_state_prompt(item, max_history=60)
        t0 = time.time()
        usage, cost, call_err = {}, 0.0, None
        try:
            raw, usage, cost = call_llm(model_id, SYSTEM_STATE, prompt_s,
                                        litellm_mod, max_tokens=max_tokens,
                                        thinking_mode=thinking_mode,
                                        reasoning_effort=reasoning_effort)
            dt = time.time() - t0
            parsed = parse_json_loose(raw)
            _bill(usage, cost)
        except Exception as e:
            raw = f"ERR:{e}"; parsed = None; dt = time.time() - t0
            call_err = str(e)[:200]
        probe_v1 = grade_state_probe(parsed, item["final_state_cf"])

        # v2: testbed-level predeclared paths + null-baseline subtraction
        probe_v2 = None
        domain_mod = TESTBEDS.get(item["domain"])
        probe_spec = getattr(domain_mod, "PROBE_PATHS_V2", None)
        if probe_spec and item.get("final_state_original") is not None:
            probe_v2 = grade_state_probe_v2(parsed,
                                            item["final_state_cf"],
                                            item["final_state_original"],
                                            probe_spec)

        phase_c = {"raw": raw[:160], "latency_s": round(dt, 2),
                   "valid": probe_v1.get("valid", False),
                   "n_paths": probe_v1.get("n_paths", 0),
                   "n_match": probe_v1.get("n_match", 0),
                   "fraction_match": probe_v1.get("fraction_match", 0.0),
                   "v2": probe_v2}
        _emit_call("C", None, "state_probe", dt, usage, cost,
                   predicted=parsed, error=call_err)

    # --- Summary for this item
    suffix_n = len(phase_a)
    suffix_wf = sum(1 for s in phase_a if s["well_formed"])
    suffix_vm = sum(1 for s in phase_a if s["value_match"])
    suffix_sm = sum(1 for s in phase_a if s["status_match"])
    first_div = next((s["suffix_idx"] for s in phase_a
                      if not s["value_match"] and s["well_formed"]), None)
    return {
        "item_id": item.get("item_id"),
        "domain": item["domain"],
        "intervention_kind": kind,
        "prefix_len": item.get("prefix_len"),
        "suffix_len": suffix_n,
        "n_well_formed": suffix_wf,
        "n_value_match": suffix_vm,
        "n_status_match": suffix_sm,
        "first_value_mismatch_at": first_div,
        "expected_delay": item.get("expected_delay"),
        "billing": bill,
        "phase_a": phase_a,
        "phase_b": phase_b,
        "phase_c": phase_c,
    }


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-jsonl", required=True,
                    help="counterfactual items jsonl from gen_counterfactuals.py")
    ap.add_argument("--model", default="bedrock/moonshotai.kimi-k2.5")
    ap.add_argument("--strategy", choices=["full_history", "sliding", "prefix_only", "free_rollout"],
                    default="sliding")
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--max-items", type=int, default=50)
    ap.add_argument("--suffix-budget", type=int, default=40)
    ap.add_argument("--max-concurrency", type=int, default=4)
    ap.add_argument("--no-explain", action="store_true")
    ap.add_argument("--no-state",   action="store_true")
    ap.add_argument("--thinking", choices=["off", "on"], default="off",
                    help="Enable provider-side thinking/reasoning. DS Flash is "
                         "always off regardless of this flag.")
    ap.add_argument("--reasoning-effort", choices=["low", "medium", "high"],
                    default=os.environ.get("CTE_REASONING_EFFORT", "high"),
                    help="Provider-side reasoning effort for thinking runs "
                         "when the provider supports it.")
    ap.add_argument("--max-tokens", type=int, default=4000,
                    help="max_tokens for counterfactual + state-probe calls. "
                         "Thinking models need 12000-20000 so they can finish "
                         "reasoning AND emit the content JSON.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--calls-dir", default=None,
                    help="If set, stream every LLM call as a jsonl line to "
                         "<calls-dir>/calls.jsonl as soon as it finishes, "
                         "before the enclosing item completes. Useful for "
                         "long-thinking models where an item might take tens "
                         "of minutes to finish.")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import litellm
    litellm.drop_params = True
    litellm.suppress_debug_info = True

    # Load items
    items: list = []
    with open(args.cf_jsonl) as f:
        for line in f:
            items.append(json.loads(line))
            if len(items) >= args.max_items:
                break
    if not items:
        print("no items to evaluate")
        return

    # Checkpoint support: every N items we flush partial results next to the
    # output path. On restart we read them back in and skip already-done items.
    # The ckpt JSONL is keyed by item_id so restart order doesn't matter.
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.with_suffix(out_path.suffix + ".ckpt.jsonl")
    done_by_id: dict = {}
    if ckpt_path.exists():
        with ckpt_path.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("item_id"):
                    done_by_id[rec["item_id"]] = rec
        print(f"[resume] loaded {len(done_by_id)} items from {ckpt_path.name}")

    print(f"evaluating {len(items)} items with model={args.model} "
          f"strategy={args.strategy} window={args.window}")

    total_bill = {"prompt_tokens": 0, "completion_tokens": 0,
                  "reasoning_tokens": 0, "cached_tokens": 0,
                  "cost_usd": 0.0, "n_calls": 0}

    results: list = [None] * len(items)
    # Pre-fill results with checkpoint hits and accumulate their billing.
    pending_idxs = []
    for i, it in enumerate(items):
        iid = it.get("item_id")
        if iid and iid in done_by_id:
            results[i] = done_by_id[iid]
            b = results[i].get("billing", {}) or {}
            for k in total_bill:
                total_bill[k] += b.get(k, 0) or 0
        else:
            pending_idxs.append(i)
    print(f"[resume] {len(pending_idxs)} items pending "
          f"(restored total=${total_bill['cost_usd']:.2f})")

    # Open checkpoint log in append mode; fsync after each write so a kill
    # can't lose the last item.
    ckpt_f = ckpt_path.open("a")

    # Per-call streaming logger: one JSON line per LLM call, flushed + fsync'd
    # the moment the call returns. This is independent of item-level ckpt
    # writes — useful for long-thinking models where one item may take tens
    # of minutes to finish.
    calls_lock = threading.Lock()
    calls_f = None
    if args.calls_dir:
        calls_dir = Path(args.calls_dir)
        calls_dir.mkdir(parents=True, exist_ok=True)
        calls_f = (calls_dir / "calls.jsonl").open("a")
        print(f"[calls-dir] streaming every call to {calls_dir/'calls.jsonl'}")

    def per_call_logger(rec: dict) -> None:
        if calls_f is None:
            return
        with calls_lock:
            calls_f.write(json.dumps(rec, default=str) + "\n")
            calls_f.flush()
            try:
                os.fsync(calls_f.fileno())
            except OSError:
                pass

    with ThreadPoolExecutor(max_workers=args.max_concurrency) as pool:
        futs = {}
        for i in pending_idxs:
            it = items[i]
            futs[pool.submit(eval_item, it, args.model, args.strategy,
                             args.window, litellm, args.suffix_budget,
                             not args.no_explain, not args.no_state,
                             args.thinking, args.max_tokens,
                             args.reasoning_effort,
                             per_call_logger)] = i
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
                r = results[i]
                b = r.get("billing", {}) or {}
                for k in total_bill:
                    total_bill[k] += b.get(k, 0) or 0
                # checkpoint: append this item to the ckpt jsonl and flush
                ckpt_f.write(json.dumps(r, default=str) + "\n")
                ckpt_f.flush()
                try:
                    os.fsync(ckpt_f.fileno())
                except OSError:
                    pass
                in_tok = b.get('prompt_tokens', 0) or 0
                cached = b.get('cached_tokens', 0) or 0
                hit_pct = (cached / in_tok * 100) if in_tok else 0.0
                cum_in = total_bill['prompt_tokens'] or 0
                cum_cached = total_bill['cached_tokens'] or 0
                cum_hit = (cum_cached / cum_in * 100) if cum_in else 0.0
                print(f"  item {i:3d} [{r['domain']:<12} {r['intervention_kind']:<18}] "
                      f"vm={r['n_value_match']}/{r['suffix_len']} "
                      f"div@{r['first_value_mismatch_at']} "
                      f"${b.get('cost_usd',0):.3f} "
                      f"(in={in_tok} cache={cached} [{hit_pct:.0f}%] "
                      f"out={b.get('completion_tokens',0)} "
                      f"think={b.get('reasoning_tokens',0)}) "
                      f"| total=${total_bill['cost_usd']:.2f} "
                      f"cum_cache={cum_hit:.0f}%",
                      flush=True)
            except Exception as e:
                print(f"  item {i} FAILED: {e}", flush=True)
                results[i] = {"item_id": items[i].get("item_id"), "error": str(e)}

    ckpt_f.close()
    if calls_f is not None:
        calls_f.close()

    # Aggregate by (domain, kind) and by prefix-length bucket
    agg_dk = defaultdict(lambda: {"n": 0, "suffix": 0, "vm": 0, "sm": 0,
                                  "wf": 0, "first_div_sum": 0, "first_div_n": 0,
                                  "explain_exact": 0, "explain_adjacent": 0,
                                  "explain_applicable": 0,
                                  "state_paths": 0, "state_match": 0,
                                  "state_n": 0})
    for r in results:
        if r is None or "error" in r:
            continue
        key = f"{r['domain']}::{r['intervention_kind']}"
        a = agg_dk[key]
        a["n"] += 1
        a["suffix"] += r["suffix_len"]
        a["vm"] += r["n_value_match"]; a["sm"] += r["n_status_match"]
        a["wf"] += r["n_well_formed"]
        if r["first_value_mismatch_at"] is not None:
            a["first_div_sum"] += r["first_value_mismatch_at"]
            a["first_div_n"] += 1
        if r.get("phase_b") and r["phase_b"].get("applicable"):
            a["explain_applicable"] += 1
            if r["phase_b"].get("exact_match"):    a["explain_exact"] += 1
            if r["phase_b"].get("adjacent_match"): a["explain_adjacent"] += 1
        if r.get("phase_c") and r["phase_c"].get("valid"):
            a["state_n"] += 1
            a["state_paths"] += r["phase_c"]["n_paths"]
            a["state_match"] += r["phase_c"]["n_match"]

    out_path.write_text(json.dumps({
        "model": args.model,
        "strategy": args.strategy,
        "window": args.window,
        "thinking": args.thinking,
        "reasoning_effort": args.reasoning_effort,
        "max_items": args.max_items,
        "suffix_budget": args.suffix_budget,
        "n_items": len(items),
        "billing_total": total_bill,
        "per_item": results,
        "aggregate_by_domain_kind": dict(agg_dk),
    }, indent=2, default=str))
    print(f"\n=== billing total ({args.model}, thinking={args.thinking}) ===")
    print(f"  calls:     {total_bill['n_calls']}")
    print(f"  input:     {total_bill['prompt_tokens']:,} tokens "
          f"(cached {total_bill['cached_tokens']:,})")
    print(f"  output:    {total_bill['completion_tokens']:,} tokens "
          f"(reasoning {total_bill['reasoning_tokens']:,})")
    print(f"  TOTAL:     ${total_bill['cost_usd']:.4f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
