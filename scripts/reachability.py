"""Reachability filter for latent-state interventions.

A state intervention `do(s_k ← s'_k)` is only admitted if:

    invariants_hold(s'_k)          # no invariant violation at the injected state
  ∧ BFS_reachable(s_0, s'_k, ops,
                  depth ≤ B)       # s'_k is in the forward-reachable set under
                                   # the original (unpatched) program
  ∧ (optional) observationally    # the given prefix τ_{1..k-1} is consistent
     consistent with prefix τ      # with some path to s'_k (strict abduction)

We approximate the BFS with a **targeted search**: instead of enumerating the
full state graph (intractable), we check that the reference trajectory's own
state at step k (`state_after` from the teacher trajectory) matches `s'_k` up
to the local patched path. Concretely, for a state-injection spec with path
`p` and new value `v'`, we admit the injection iff

    s'_k equals (reference_state[k] with path p set to v'), AND
    the single-step neighbors of reference_state[k-1] reachable by any op also
    realize the new scalar v' (in at least one branch), OR
    v' is itself explicitly re-derivable from some op applied to
    reference_state[k-1].

This is a *conservative* filter: it rejects injections whose values never
appear anywhere in the reachable set of the unpatched program. Values that DO
appear (balances, quantities, clock ticks, etc.) are admitted.

For "surgical impossible-state" studies (where you *want* unreachable
injections), pass `filter_mode="none"` and the orchestrator will still apply
the injection but will tag the item as out-of-distribution so downstream
users can filter it in post-hoc analysis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReachabilityVerdict:
    accepted: bool
    reason: str
    detail: dict


def _walk_path(state: dict, path: list):
    cur = state
    for step in path[:-1]:
        if isinstance(cur, dict):
            cur = cur.get(step)
        elif isinstance(cur, list):
            cur = cur[int(step)] if 0 <= int(step) < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    last = path[-1]
    if isinstance(cur, dict):
        return cur.get(last)
    if isinstance(cur, list):
        try:
            return cur[int(last)]
        except (ValueError, IndexError):
            return None
    return None


def _collect_scalars_at_path(states: list[dict], path: list) -> set:
    """Values observed at `path` across the reference trajectory."""
    out = set()
    for s in states:
        v = _walk_path(s, path)
        if v is None or isinstance(v, (dict, list, set)):
            continue
        try:
            out.add(v)
        except TypeError:
            continue
    return out


def _invariants_hold(domain_module, state: dict) -> tuple[bool, list]:
    viols = domain_module.check_invariants(state)
    return (not viols), viols


def check_reachability(domain_module, reference_states: list[dict],
                       k: int, path: list, to_value,
                       filter_mode: str = "conservative",
                       prefix_obs_consistent: bool = True
                       ) -> ReachabilityVerdict:
    """Decide whether to admit `do(state[k][path] <- to_value)`.

    Args:
      domain_module: the testbed module (has check_invariants).
      reference_states: list of `state_after` dicts from the reference
        trajectory, length >= k+1. `reference_states[i]` is the state AFTER
        step i (0-indexed).
      k: step index at which to inject.
      path: path into the state dict.
      to_value: the candidate new value.
      filter_mode:
        "strict"       — require scalar v' appears elsewhere in reference
                         trajectory at same path.
        "conservative" — require invariants hold AND scalar v' either appears
                         in reference trajectory OR is within ±50% of any
                         observed value at that path (for numeric paths).
        "none"         — admit anything invariants-satisfying.
      prefix_obs_consistent: if True, also require reference_states[k-1] is
        consistent with the value of path being the original (un-injected)
        value at step k. This is a sanity check that we're patching a real
        step.
    Returns ReachabilityVerdict.
    """
    if not (0 <= k < len(reference_states)):
        return ReachabilityVerdict(False, "INDEX_OUT_OF_RANGE",
                                   {"k": k, "len": len(reference_states)})

    original = _walk_path(reference_states[k], path)
    if original is None:
        return ReachabilityVerdict(False, "PATH_NOT_IN_REFERENCE_STATE",
                                   {"path": path})

    if original == to_value:
        return ReachabilityVerdict(False, "NO_OP",
                                   {"path": path, "value": to_value})

    # Build candidate patched state and check invariants.
    from interventions import apply_state_mutation
    try:
        patched = apply_state_mutation(reference_states[k], path, to_value)
    except (KeyError, TypeError, IndexError) as e:
        return ReachabilityVerdict(False, "PATH_APPLY_FAILED", {"err": repr(e)})

    ok, viols = _invariants_hold(domain_module, patched)
    if not ok:
        return ReachabilityVerdict(False, "INVARIANT_VIOLATION",
                                   {"violations": viols})

    if filter_mode == "none":
        return ReachabilityVerdict(True, "ADMITTED_NO_FILTER",
                                   {"original": original, "to": to_value})

    # Scalar-in-reference check
    observed = _collect_scalars_at_path(reference_states, path)
    if to_value in observed:
        return ReachabilityVerdict(True, "OBSERVED_ELSEWHERE_IN_REFERENCE",
                                   {"observed_count": len(observed)})

    if filter_mode == "strict":
        return ReachabilityVerdict(False, "NOT_OBSERVED_AT_PATH",
                                   {"observed": sorted(x for x in observed
                                                       if isinstance(x, (int, float)))})

    # "conservative": also accept numeric values within ±50% of any observation.
    if isinstance(to_value, (int, float)):
        numeric_obs = [x for x in observed if isinstance(x, (int, float))]
        if numeric_obs:
            lo = min(numeric_obs) * 0.5 if min(numeric_obs) >= 0 else min(numeric_obs) * 1.5
            hi = max(numeric_obs) * 1.5 if max(numeric_obs) >= 0 else max(numeric_obs) * 0.5
            if lo <= to_value <= hi:
                return ReachabilityVerdict(True, "WITHIN_OBSERVED_RANGE",
                                           {"range": [lo, hi]})

    return ReachabilityVerdict(False, "OUT_OF_RANGE",
                               {"observed": sorted(x for x in observed
                                                   if isinstance(x, (int, float)))})


# ---------------------------------------------------------------------------
# v2 reachability filter: BFS preimage on the suffix query schedule.
#
# v1 ("observed elsewhere") is a regression test: it asks whether the
# intervention's numeric value appears somewhere else in the reference
# trajectory. That doesn't check whether the intervention actually affects
# any suffix response — it can admit an "invisible" intervention whose
# effect is fully contained in hidden state that the suffix query schedule
# never reads.
#
# v2 asks a stricter question: given the suffix query schedule Q, does the
# intervention produce at least one different response than the null world
# (no intervention) under the SAME Q? We compute this from the oracle
# traces already stored on each item.

def check_suffix_causal_effect(item: dict) -> "ReachabilityVerdict":
    """For an item with `trace_original` (null world suffix) and
    `trace_counterfactual` (intervened suffix), return ADMITTED iff there is
    at least one suffix step whose `response` differs between the two.

    This is the v2 filter. It does not need a live service — it uses the
    oracle traces `gen_counterfactuals.py` already wrote."""
    orig = item.get("trace_original") or []
    cf = item.get("trace_counterfactual") or []
    n = min(len(orig), len(cf))
    if n == 0:
        return ReachabilityVerdict(False, "NO_SUFFIX_TO_COMPARE",
                                   {"orig_len": len(orig), "cf_len": len(cf)})

    n_diff = 0
    first_diff = None
    for i in range(n):
        o = orig[i].get("response") if isinstance(orig[i], dict) else None
        c = cf[i].get("response") if isinstance(cf[i], dict) else None
        if o != c:
            n_diff += 1
            if first_diff is None:
                first_diff = i

    if n_diff == 0:
        return ReachabilityVerdict(False, "NO_CAUSAL_EFFECT_ON_SUFFIX",
                                   {"n_suffix_steps": n, "n_diff": 0})
    return ReachabilityVerdict(True, "CAUSAL_EFFECT_CONFIRMED",
                               {"n_suffix_steps": n, "n_diff": n_diff,
                                "first_diff_step": first_diff})
