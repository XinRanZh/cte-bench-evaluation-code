"""Counterfactual trace generator — orchestrator.

Takes reference trajectories produced by `gen_trajectories.py` and, for each
trajectory, produces one or more counterfactual items:

    item = {
        "domain":              str,
        "item_id":              str,   # stable hash
        "intervention":         {kind, spec_dict...},
        "explanation_truth":    {line_number, original_span, modified_span},
        "prefix_len":           int k,
        "trace_prefix":         [(op, args, response)] length k,
        "trace_counterfactual": [(op, args, response_cf, state_after_cf)] length n-k,
        "trace_original":       [(op, args, response_orig)] length n-k,
        "reachability":         {verdict, reason},
        "final_violations_cf":  [...],
        "expected_delay":       int | None,  # adversarial hard-subset metadata
    }

Intervention families produced (per reference trajectory, quotas configurable):
  - threshold_change  (AST-based numeric perturbation)
  - branch_inversion  (AST-based comparator flip)
  - quota_change      (module-level constant change; reservation only)
  - state_injection   (local state overwrite, reachability-filtered)
  - clock_jump        (state_injection on clock/advance, filtered)
  - adversarial       (hand-designed delayed-effect patches)

Usage:
    python scripts/gen_counterfactuals.py \
        --in-dir data/ \
        --out-dir counterfactuals/ \
        --domains bank cart auth reservation ratelimiter filesystem \
        --split eval \
        --per-traj 5 \
        --prefix-fracs 0.10 0.30 0.50 0.70 \
        --seed 42
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, random, sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from testbeds import TESTBEDS
from interventions import (
    apply_threshold_change, apply_branch_inversion, apply_quota_change,
    apply_state_mutation, load_service_from_source, DOMAIN_CLASS,
)
from reachability import check_reachability
from adversarial_patches import PATCHES, apply_adversarial_patch


# ---------------------------------------------------------------------------
# Live-object state mutation (inverse of state_dict() by attribute reflection).

def _mutate_attr(svc, path: list, to_value):
    """Mutate `svc.<path[0]>[path[1]]...[path[-1]] = to_value` in-place.

    Path[0] is expected to be a top-level attribute of the service object
    (matches state_dict() keys by convention across all 6 testbeds).
    """
    if not path:
        raise ValueError("empty path")
    attr = getattr(svc, path[0])
    for step in path[1:-1]:
        if isinstance(attr, dict):
            attr = attr[step]
        elif isinstance(attr, list):
            attr = attr[int(step)]
        else:
            raise TypeError(f"cannot descend into {type(attr).__name__}")
    last = path[-1]
    if len(path) == 1:
        setattr(svc, last, to_value)
        return
    if isinstance(attr, dict):
        attr[last] = to_value
    elif isinstance(attr, list):
        attr[int(last)] = to_value
    else:
        # single-level path: setattr(svc, last, to_value) handled above.
        raise TypeError(f"cannot assign into {type(attr).__name__}")


# ---------------------------------------------------------------------------
# Service construction: either from original module or from patched source.

def _service_factory(domain: str, patched_source: Optional[str] = None):
    """Return a callable that produces a service instance.

    For patched sources we also graft `state_dict` from the reference class
    onto the patched class. The reference module keeps `state_dict` and
    invariant-checking machinery OUT of SOURCE_CODE on purpose (we don't
    want to leak hidden state into the LLM prompt). So the exec'd patched
    class is missing it and we re-add it here.
    """
    mod = TESTBEDS[domain]
    if patched_source is None:
        return mod.new_service
    cls = load_service_from_source(patched_source, DOMAIN_CLASS[domain])
    ref_cls = type(mod.new_service())
    if not hasattr(cls, "state_dict"):
        # Bind the reference class's state_dict onto the patched class. The
        # method reads self.<attr>, and both classes share the same attribute
        # schema by construction.
        cls.state_dict = ref_cls.state_dict
    def make():
        try:
            return cls()
        except TypeError:
            ref = mod.new_service()
            kw = {}
            for k in ("limit", "expiry"):
                if hasattr(ref, k):
                    kw[k] = getattr(ref, k)
            return cls(**kw)
    return make


def _replay(svc, steps: list, stop_at: Optional[int] = None):
    """Execute ops from steps onto svc; return list of (op, args, response).

    If stop_at is given, replay steps[0:stop_at] only. Exceptions raised by
    the (possibly patched) service become `{"error":{"code":"SERVICE_EXCEPTION",
    "message": repr(exc)}}` responses — this is legitimate ground truth for
    counterfactual programs that crash on inputs that never reach that path
    in the original.
    """
    out = []
    end = len(steps) if stop_at is None else stop_at
    for s in steps[:end]:
        op = s["op"]; args = s["args"]
        method = getattr(svc, op, None)
        if method is None:
            resp = {"error": {"code": "OP_MISSING", "message": op}}
            out.append({"op": op, "args": args, "response": resp})
            continue
        try:
            try:
                resp = method(**args)
            except TypeError:
                resp = method(*list(args.values()))
        except Exception as exc:
            resp = {"error": {"code": "SERVICE_EXCEPTION",
                              "message": f"{type(exc).__name__}: {exc}"}}
        out.append({"op": op, "args": args, "response": resp})
    return out


# ---------------------------------------------------------------------------
# Intervention builders.

def _code_intervention_variants(domain: str, source: str, rng: random.Random):
    """Enumerate code-intervention candidates as lightweight specs."""
    out = []
    # Threshold
    from interventions import find_threshold_candidates
    cands = find_threshold_candidates(source)
    for i in range(len(cands)):
        out.append(("threshold_change", {"choice_index": i}))
    # Branch
    from interventions import find_branch_candidates
    cands = find_branch_candidates(source)
    for i in range(len(cands)):
        out.append(("branch_inversion", {"choice_index": i}))
    # Quota
    from interventions import find_quota_candidates
    cands = find_quota_candidates(source)
    for i in range(len(cands)):
        out.append(("quota_change", {"choice_index": i}))
    rng.shuffle(out)
    return out


def _state_injection_candidates(domain: str, trajectory: dict, k: int, rng: random.Random):
    """Return a list of (path, to_value) candidates reading from the
    reference trajectory's observed value distribution at each scalar path.
    """
    mod = TESTBEDS[domain]
    states = [s["state_after"] for s in trajectory["steps"]]
    state_k = states[k]
    scalar_paths = _enumerate_scalar_paths(state_k)
    # Collect observed values per path across all states.
    observed: dict[tuple, set] = {}
    for s in states:
        for p in scalar_paths:
            v = _deep_get(s, p)
            if isinstance(v, (int, float, str, bool)):
                observed.setdefault(p, set()).add(v)
    cands = []
    for path, vals in observed.items():
        orig = _deep_get(state_k, list(path))
        for v in vals:
            if v != orig:
                cands.append((list(path), v))
    rng.shuffle(cands)
    return cands


def _enumerate_scalar_paths(obj, prefix: Optional[list] = None, out: Optional[list] = None):
    if prefix is None: prefix = []
    if out is None: out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            _enumerate_scalar_paths(v, prefix + [k], out)
    elif isinstance(obj, list):
        # enumerate only indexed scalars; for our state_dicts lists are small or nested dicts.
        for i, v in enumerate(obj):
            _enumerate_scalar_paths(v, prefix + [i], out)
    else:
        out.append(tuple(prefix))
    return out


def _deep_get(obj, path: list):
    cur = obj
    for step in path:
        if isinstance(cur, dict):
            cur = cur.get(step)
        elif isinstance(cur, list):
            if isinstance(step, int) and 0 <= step < len(cur):
                cur = cur[step]
            else:
                return None
        else:
            return None
    return cur


# ---------------------------------------------------------------------------
# Per-item generators.

def _make_id(*parts) -> str:
    return hashlib.sha256("::".join(map(str, parts)).encode()).hexdigest()[:12]


def generate_code_item(domain: str, trajectory: dict, k: int,
                       kind: str, kwargs: dict) -> Optional[dict]:
    src = trajectory["source_code"]
    fn = {"threshold_change": apply_threshold_change,
          "branch_inversion": apply_branch_inversion,
          "quota_change":     apply_quota_change}[kind]
    try:
        patched_source, spec = fn(src, domain, **kwargs)
    except ValueError:
        return None

    # Pearl do-calculus semantics: abduction on the ORIGINAL program using the
    # observed prefix to fix the latent state at step k-1, then intervention
    # `do(P <- P')` from step k onward. We realize this by replaying the
    # prefix on the original service, then swapping the service's class to
    # the patched one before replaying the suffix. Attribute dict is shared,
    # only method dispatch changes.
    orig_factory = _service_factory(domain)
    svc = orig_factory()
    _replay(svc, trajectory["steps"], stop_at=k)
    patched_cls = load_service_from_source(patched_source, DOMAIN_CLASS[domain])
    ref_cls = type(orig_factory())
    if not hasattr(patched_cls, "state_dict"):
        patched_cls.state_dict = ref_cls.state_dict
    svc.__class__ = patched_cls
    cf_suffix = _replay(svc, trajectory["steps"][k:])

    mod = TESTBEDS[domain]
    try:
        final_state_cf = svc.state_dict()
        cf_viols = mod.check_invariants(final_state_cf)
    except Exception as exc:
        final_state_cf = None
        cf_viols = [f"STATE_DICT_EXCEPTION:{type(exc).__name__}"]

    return {
        "domain": domain,
        "item_id": _make_id(domain, kind, k, trajectory["steps"][0]["args"],
                            spec.line_number, spec.original_span, spec.modified_span),
        "intervention": {"kind": kind,
                         "line_number": spec.line_number,
                         "col_offset": spec.col_offset,
                         "original_span": spec.original_span,
                         "modified_span": spec.modified_span,
                         "description": spec.description},
        "explanation_truth": {"line_number": spec.line_number,
                              "original_span": spec.original_span,
                              "modified_span": spec.modified_span},
        "prefix_len": k,
        "trace_prefix": [{"op": s["op"], "args": s["args"], "response": s["response"]}
                         for s in trajectory["steps"][:k]],
        "trace_original": [{"op": s["op"], "args": s["args"], "response": s["response"]}
                           for s in trajectory["steps"][k:]],
        "trace_counterfactual": cf_suffix,
        "patched_source": patched_source,
        "reachability": {"accepted": True, "reason": "CODE_INTERVENTION"},
        "final_state_cf": final_state_cf,
        "final_state_original": trajectory["steps"][-1].get("state_after"),
        "final_violations_cf": cf_viols,
        "expected_delay": None,
    }


def generate_state_item(domain: str, trajectory: dict, k: int,
                        path: list, to_value) -> Optional[dict]:
    mod = TESTBEDS[domain]
    states = [s["state_after"] for s in trajectory["steps"]]
    verdict = check_reachability(mod, states, k, path, to_value,
                                 filter_mode="conservative")
    if not verdict.accepted:
        return None

    # Re-execute τ_{0..k} on a fresh service, then mutate live state, then run τ_{k+1..n}.
    factory = _service_factory(domain)
    svc = factory()
    _replay(svc, trajectory["steps"], stop_at=k + 1)
    try:
        _mutate_attr(svc, path, to_value)
    except Exception:
        return None
    cf_suffix = _replay(svc, trajectory["steps"][k + 1:])

    try:
        final_state_cf = svc.state_dict()
        cf_viols = mod.check_invariants(final_state_cf)
    except Exception as exc:
        final_state_cf = None
        cf_viols = [f"STATE_DICT_EXCEPTION:{type(exc).__name__}"]

    from_value = _deep_get(states[k], path)
    kind = "clock_jump" if path and path[-1] == "clock" else "state_injection"

    return {
        "domain": domain,
        "item_id": _make_id(domain, kind, k, path, to_value),
        "intervention": {"kind": kind,
                         "path": path,
                         "from_value": from_value,
                         "to_value": to_value,
                         "description": f"state[{'.'.join(map(str, path))}] "
                                        f"{from_value!r} -> {to_value!r} at step {k}"},
        "explanation_truth": {"path": path,
                              "from_value": from_value,
                              "to_value": to_value},
        "prefix_len": k + 1,
        "trace_prefix": [{"op": s["op"], "args": s["args"], "response": s["response"]}
                         for s in trajectory["steps"][:k + 1]],
        "trace_original": [{"op": s["op"], "args": s["args"], "response": s["response"]}
                           for s in trajectory["steps"][k + 1:]],
        "trace_counterfactual": cf_suffix,
        "patched_source": None,
        "reachability": {"accepted": True, "reason": verdict.reason,
                         "detail": verdict.detail},
        "final_state_cf": final_state_cf,
        "final_state_original": trajectory["steps"][-1].get("state_after"),
        "final_violations_cf": cf_viols,
        "expected_delay": None,
    }


def generate_adversarial_item(domain: str, trajectory: dict, k: int,
                              patch) -> Optional[dict]:
    src = trajectory["source_code"]
    try:
        patched_source, spec = apply_adversarial_patch(src, patch)
    except Exception:
        return None
    # Do-calculus: prefix replay on ORIGINAL program, swap class at step k.
    orig_factory = _service_factory(domain)
    svc = orig_factory()
    _replay(svc, trajectory["steps"], stop_at=k)
    try:
        patched_cls = load_service_from_source(patched_source, DOMAIN_CLASS[domain])
    except Exception:
        return None
    ref_cls = type(orig_factory())
    if not hasattr(patched_cls, "state_dict"):
        patched_cls.state_dict = ref_cls.state_dict
    svc.__class__ = patched_cls
    cf_suffix = _replay(svc, trajectory["steps"][k:])
    mod = TESTBEDS[domain]
    try:
        final_state_cf = svc.state_dict()
        cf_viols = mod.check_invariants(final_state_cf)
    except Exception as exc:
        final_state_cf = None
        cf_viols = [f"STATE_DICT_EXCEPTION:{type(exc).__name__}"]
    return {
        "domain": domain,
        "item_id": _make_id(domain, "adversarial", patch.name, k),
        "intervention": {"kind": "adversarial",
                         "name": spec["name"],
                         "line_number": spec["line_number"],
                         "col_offset": spec["col_offset"],
                         "original_span": spec["original_span"],
                         "modified_span": spec["modified_span"],
                         "description": spec["description"]},
        "explanation_truth": {"line_number": spec["line_number"],
                              "original_span": spec["original_span"],
                              "modified_span": spec["modified_span"]},
        "prefix_len": k,
        "trace_prefix": [{"op": s["op"], "args": s["args"], "response": s["response"]}
                         for s in trajectory["steps"][:k]],
        "trace_original": [{"op": s["op"], "args": s["args"], "response": s["response"]}
                           for s in trajectory["steps"][k:]],
        "trace_counterfactual": cf_suffix,
        "patched_source": patched_source,
        "reachability": {"accepted": True, "reason": "ADVERSARIAL"},
        "final_state_cf": final_state_cf,
        "final_state_original": trajectory["steps"][-1].get("state_after"),
        "final_violations_cf": cf_viols,
        "expected_delay": spec["expected_delay"],
    }


# ---------------------------------------------------------------------------
# Main.

def _response_diverges(orig_resp: list, cf_resp: list) -> bool:
    """True iff the counterfactual response trace differs from the original
    in at least one step. We compare responses only (args/ops are identical)."""
    for a, b in zip(orig_resp, cf_resp):
        if a["response"] != b["response"]:
            return True
    return False


def generate_for_trajectory(domain: str, trajectory: dict, per_type: dict,
                            prefix_fracs: list, rng: random.Random
                            ) -> list[dict]:
    n_steps = len(trajectory["steps"])
    k_options = sorted({max(1, min(n_steps - 1, int(n_steps * f)))
                        for f in prefix_fracs})
    items = []
    src = trajectory["source_code"]

    # --- code interventions ---
    variants = _code_intervention_variants(domain, src, rng)
    attempts = 0
    for kind, kwargs in variants:
        if per_type.get(kind, 0) <= 0:
            continue
        k = rng.choice(k_options)
        item = generate_code_item(domain, trajectory, k, kind, kwargs)
        attempts += 1
        if item is None:
            continue
        # Reject items whose counterfactual trace is byte-identical to the
        # original — the intervention is a no-op on this trajectory.
        if not _response_diverges(item["trace_original"], item["trace_counterfactual"]):
            continue
        items.append(item)
        per_type[kind] -= 1

    # --- state interventions ---
    if per_type.get("state_injection", 0) > 0 or per_type.get("clock_jump", 0) > 0:
        cands = _state_injection_candidates(domain, trajectory, k_options[0], rng)
        for path, to_value in cands:
            which = "clock_jump" if path and path[-1] == "clock" else "state_injection"
            if per_type.get(which, 0) <= 0:
                continue
            k = rng.choice(k_options)
            item = generate_state_item(domain, trajectory, k, path, to_value)
            if item is None:
                continue
            if not _response_diverges(item["trace_original"], item["trace_counterfactual"]):
                continue
            items.append(item)
            per_type[which] -= 1
            if all(v <= 0 for k2, v in per_type.items() if k2 in ("state_injection", "clock_jump")):
                break

    # --- adversarial ---
    # Keep k SMALL for adversarial patches so the suffix has room to trigger
    # the delayed effect. We prefer the 10-30% quartile of the trajectory.
    if per_type.get("adversarial", 0) > 0:
        adv_k_options = sorted({max(1, min(n_steps - 1, int(n_steps * f)))
                                for f in prefix_fracs if f <= 0.30})
        if not adv_k_options:
            adv_k_options = [max(1, n_steps // 10)]
        for patch in PATCHES:
            if patch.domain != domain:
                continue
            if per_type["adversarial"] <= 0:
                break
            # Try each candidate k; accept the first that yields divergence
            # (or the last attempt if none divergent — the item is still
            # valuable as a "rare latent difference" example).
            chosen = None
            for k in adv_k_options:
                item = generate_adversarial_item(domain, trajectory, k, patch)
                if item is None:
                    continue
                if _response_diverges(item["trace_original"], item["trace_counterfactual"]):
                    chosen = item
                    break
                chosen = item  # retain last successful exec even if no divergence
            if chosen is not None:
                items.append(chosen)
                per_type["adversarial"] -= 1

    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",  default="data",
                    help="directory with {domain}/{split}.jsonl from gen_trajectories.py")
    ap.add_argument("--out-dir", default="counterfactuals")
    ap.add_argument("--domains", nargs="+",
                    default=["bank", "cart", "auth", "reservation", "ratelimiter", "filesystem"])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--per-traj", type=int, default=6,
                    help="approx total counterfactual items per reference trajectory")
    ap.add_argument("--prefix-fracs", type=float, nargs="+",
                    default=[0.10, 0.30, 0.50, 0.70])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-adversarial", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # Default per-type quotas within each trajectory (sums to ~args.per_traj).
    base_quotas = {
        "threshold_change": 1,
        "branch_inversion": 1,
        "quota_change":     0,       # only reservation has candidates; bumped below
        "state_injection":  2,
        "clock_jump":       1,
        "adversarial":      1 if args.include_adversarial else 0,
    }

    for domain in args.domains:
        in_path = Path(args.in_dir) / domain / f"{args.split}.jsonl"
        if not in_path.exists():
            print(f"  skip {domain}: {in_path} not found")
            continue
        dom_out = out_dir / f"{domain}.jsonl"
        n_items = 0
        stats: dict[str, int] = {}
        with in_path.open() as f_in, dom_out.open("w") as f_out:
            for line in f_in:
                traj = json.loads(line)
                quotas = dict(base_quotas)
                if domain == "reservation":
                    quotas["quota_change"] = 1
                items = generate_for_trajectory(domain, traj, quotas, args.prefix_fracs, rng)
                for it in items:
                    f_out.write(json.dumps(it) + "\n")
                    n_items += 1
                    k = it["intervention"]["kind"]
                    stats[k] = stats.get(k, 0) + 1
        print(f"  {domain:<12}  items={n_items:<4}  by_kind={stats}  -> {dom_out}")


if __name__ == "__main__":
    main()
