"""Generate teacher trajectories by executing the REAL service.

This is the cheap/deterministic route: no LLM needed for teacher labels, since the
REAL service is source code we can just run. Each trajectory is a sequence of
(request, response, hidden_state_after) tuples.

Why this matters for development:
  - The development question is "does a learned compressor + invariant loss beat
    a full-history LLM baseline on long-horizon consistency?"
  - For TRAINING the SCST, we need supervision: (request, prior_state, response).
    The real service gives us that exactly, for free.
  - For EVAL (measuring generalization), we compare SCST output to the real
    service output on unseen trajectories.

The LLM baseline (Kimi / Sonnet, via litellm) is only needed to run the comparison
at eval time — as a competing "simulator" without access to ground-truth state.

Saves: data/{bank,cart,auth}/trajectories.jsonl (one JSON object per line)
"""
from __future__ import annotations
import argparse, json, random, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from testbeds import TESTBEDS


def _valid_ids(svc, kind):
    """Return currently valid IDs of a given kind to prefer sampling from."""
    if kind == "account":
        return list(getattr(svc, "accounts", {}).keys())
    if kind == "user_cart":
        return list(getattr(svc, "carts", {}).keys())
    if kind == "token":
        toks = getattr(svc, "tokens", {})
        return [t for t, v in toks.items() if v.get("alive")]
    if kind == "seat_class":
        return list(getattr(svc, "capacity", {}).keys())
    if kind == "active_hold":
        return [hid for hid, h in getattr(svc, "holds", {}).items()
                if h.get("state") == "active"]
    if kind == "inode_dir":
        return [iid for iid, n in getattr(svc, "inodes", {}).items()
                if n.get("kind") == "dir"]
    if kind == "inode_file":
        return [iid for iid, n in getattr(svc, "inodes", {}).items()
                if n.get("kind") == "file"]
    if kind == "inode_any":
        return list(getattr(svc, "inodes", {}).keys())
    if kind == "locked_inode":
        return list(getattr(svc, "locks", {}).keys())
    return []


def sample_args(op_name, arg_names, rng: random.Random, svc, phase="mid", op_name_ctx=None):
    op_name = op_name_ctx or op_name  # keep backward-compat
    """Phase-aware, valid-ID-preferring sampler. Fix 2 from DEVELOPMENT_NOTES.

    phase ∈ {"early", "mid", "late"} — affects probability of invalid / error-triggering args.

    Philosophy:
      - In early phase, drive state creation (bias create_*, issue_token, add_item with new ids).
      - In mid phase, prefer operations on existing state (use valid IDs with ~85% prob).
      - In late phase, mix in some error-triggers (~25%) to maintain error coverage.
    """
    err_prob = {"early": 0.10, "mid": 0.18, "late": 0.25}[phase]
    valid_pref = {"early": 0.30, "mid": 0.85, "late": 0.75}[phase]

    valid_accts = _valid_ids(svc, "account")
    valid_toks  = _valid_ids(svc, "token")
    valid_users = _valid_ids(svc, "user_cart") or [f"u{i}" for i in range(3)]
    valid_seats = _valid_ids(svc, "seat_class")
    active_holds = _valid_ids(svc, "active_hold")
    all_holds = list(getattr(svc, "holds", {}).keys())
    dir_inodes  = _valid_ids(svc, "inode_dir") or ["i0"]
    file_inodes = _valid_ids(svc, "inode_file")
    any_inodes  = _valid_ids(svc, "inode_any") or ["i0"]
    locked_inodes = _valid_ids(svc, "locked_inode")

    def uid():
        # bias toward a small stable pool of users
        if valid_users and rng.random() < valid_pref:
            return rng.choice(valid_users)
        return f"u{rng.randint(0, 4)}"
    def pid(): return f"p{rng.randint(0, 9)}"
    def acct_existing_or_new():
        if valid_accts and rng.random() < valid_pref:
            return rng.choice(valid_accts)
        return f"a{rng.randint(0, 5)}"

    def acct_new_preferred():
        """For create_account: prefer an ID that doesn't exist yet."""
        closed = getattr(svc, "closed", set())
        used = set(valid_accts) | set(closed)
        candidates = [f"a{i}" for i in range(8) if f"a{i}" not in used]
        if candidates:
            return rng.choice(candidates)
        # All used — fallback to random (will usually fail with ACCOUNT_EXISTS, rare)
        return f"a{rng.randint(0, 7)}"
    def amt():
        # Keep amounts small (1-30) so deposits/withdrawals/transfers succeed most of the time
        # — initial accounts are seeded with 10-200, so 1-30 rarely overdraws.
        if rng.random() < err_prob:
            return round(rng.uniform(500, 5000), 2)
        return round(rng.uniform(1, 20), 2)
    def qty():
        r = rng.random()
        if r < err_prob:
            return rng.choice([0, -1, 15, 20])  # invalid
        return rng.randint(1, 10)
    def token_id():
        if valid_toks and rng.random() < valid_pref:
            return rng.choice(valid_toks)
        # occasionally reference a revoked/expired one too
        all_toks = list(getattr(svc, "tokens", {}).keys())
        if all_toks and rng.random() < 0.4:
            return rng.choice(all_toks)
        return f"t{rng.randint(1, 30)}"

    args = {}
    # Pre-pick transfer accounts to try to keep them distinct + valid
    transfer_from, transfer_to = None, None
    if "from_id" in arg_names and "to_id" in arg_names and len(valid_accts) >= 2:
        pair = rng.sample(valid_accts, 2)
        transfer_from, transfer_to = pair[0], pair[1]

    for a in arg_names:
        if a == "user_id":      args[a] = uid()
        elif a == "product_id": args[a] = pid()
        elif a == "qty":        args[a] = qty()
        elif a == "account_id":
            # If the op is create_account, prefer a NEW ID. Otherwise prefer EXISTING.
            if op_name == "create_account":
                args[a] = acct_new_preferred()
            else:
                args[a] = acct_existing_or_new()
        elif a == "from_id":    args[a] = transfer_from if transfer_from else acct_existing_or_new()
        elif a == "to_id":      args[a] = transfer_to if transfer_to else acct_existing_or_new()
        elif a == "initial":    args[a] = round(rng.uniform(10, 200), 2)
        elif a == "amount":
            # bank uses floats; ratelimiter uses ints. dispatch by op context.
            if op_name in ("hit_fixed", "hit_sliding"):
                if rng.random() < err_prob:
                    args[a] = rng.choice([0, -1, 8])
                else:
                    args[a] = rng.randint(1, 3)
            else:
                args[a] = amt()
        elif a == "token_id":   args[a] = token_id()
        elif a == "as_user_id":
            # Prefer the token's actual owner (to trigger successful access checks)
            tid = args.get("token_id")
            toks = getattr(svc, "tokens", {})
            if tid in toks and rng.random() < valid_pref:
                args[a] = toks[tid]["user_id"]
            else:
                args[a] = uid()
        elif a == "ttl":        args[a] = rng.randint(2, 15)
        elif a == "delta":      args[a] = rng.randint(0, 3)
        # --- reservation ---
        elif a == "seat_class_id":
            if op_name == "open_class":
                used = set(valid_seats)
                candidates = [c for c in ["A", "B", "C", "D"] if c not in used]
                args[a] = rng.choice(candidates) if candidates else rng.choice(["A", "B", "C", "D"])
            else:
                if valid_seats and rng.random() < valid_pref:
                    args[a] = rng.choice(valid_seats)
                else:
                    args[a] = rng.choice(["A", "B", "C", "D"])
        elif a == "capacity":   args[a] = rng.randint(3, 10)
        elif a == "hold_id":
            pool = active_holds if (active_holds and rng.random() < valid_pref) else all_holds
            args[a] = rng.choice(pool) if pool else f"h{rng.randint(1, 50)}"
        # --- ratelimiter ---
        elif a == "key":        args[a] = f"k{rng.randint(0, 3)}"
        elif a == "strategy":   args[a] = rng.choice(["fixed", "sliding"])
        # --- filesystem ---
        elif a == "parent_id":
            args[a] = rng.choice(dir_inodes) if dir_inodes else "i0"
        elif a == "name":       args[a] = f"n{rng.randint(0, 9)}"
        elif a == "mode":       args[a] = rng.choice(["w", "r"])
        elif a == "inode_id":
            if op_name in ("unlock",):
                pool = locked_inodes if (locked_inodes and rng.random() < valid_pref) else any_inodes
            elif op_name in ("write_bytes",):
                pool = file_inodes if (file_inodes and rng.random() < valid_pref) else any_inodes
            elif op_name in ("listdir",):
                pool = dir_inodes if (dir_inodes and rng.random() < valid_pref) else any_inodes
            else:
                pool = any_inodes
            args[a] = rng.choice(pool) if pool else "i0"
        elif a == "holder":     args[a] = f"sess{rng.randint(0, 3)}"
        elif a == "n":          args[a] = rng.randint(0, 200)
        else: raise ValueError(f"unknown arg {a}")
    return args


def _phase_for_step(step_idx, n_steps):
    if step_idx < n_steps * 0.15: return "early"
    if step_idx < n_steps * 0.70: return "mid"
    return "late"


def _pick_op(module, svc, phase, rng):
    """Phase-biased op selection. Drive state into existence early; exercise it mid/late.

    Also gates operations that REQUIRE state: if no accounts exist, we must create.
    """
    ops = module.OPS
    names = [o[0] for o in ops]
    weights = [1.0] * len(ops)

    creators = {"create_account", "add_item", "issue_token",
                "open_class", "hold_seats", "makedir", "openbin"}
    readers  = {"get_balance", "get_cart", "check_access",
                "inspect", "get_count", "listdir"}
    destructors = {"close_account", "empty_cart", "revoke_token",
                   "cancel_hold", "confirm_hold", "remove", "reset"}

    # Hard gate: if bank has fewer than 3 accounts, force creation
    n_accts = len(getattr(svc, "accounts", {}))
    n_toks = len([t for t, v in getattr(svc, "tokens", {}).items() if v.get("alive")])
    n_seat_classes = len(getattr(svc, "capacity", {}))
    n_dirs = sum(1 for n in getattr(svc, "inodes", {}).values() if n.get("kind") == "dir")
    n_files = sum(1 for n in getattr(svc, "inodes", {}).values() if n.get("kind") == "file")

    # Saturation detection: how full is the state pool?
    bank_saturated = module.NAME == "bank" and n_accts >= 6
    auth_saturated = module.NAME == "auth" and n_toks >= 5
    res_saturated  = module.NAME == "reservation" and n_seat_classes >= 3
    fs_saturated   = module.NAME == "filesystem" and n_files >= 6

    for i, op_name in enumerate(names):
        # --- Hard gates ---
        if module.NAME == "bank" and n_accts < 3:
            if op_name != "create_account":
                weights[i] = 0.01
        if module.NAME == "auth" and n_toks < 2:
            if op_name not in {"issue_token", "advance_clock"}:
                weights[i] *= 0.1
        if module.NAME == "reservation" and n_seat_classes == 0:
            if op_name != "open_class":
                weights[i] *= 0.01
        if module.NAME == "filesystem" and n_dirs < 2:
            # only root exists; must makedir before most ops make sense
            if op_name not in {"makedir", "listdir", "advance_clock"}:
                weights[i] *= 0.1

        # If saturated, SUPPRESS further creation (would just error)
        if bank_saturated and op_name == "create_account":
            weights[i] *= 0.1
        if auth_saturated and op_name == "issue_token":
            weights[i] *= 0.3
        if res_saturated and op_name == "open_class":
            weights[i] *= 0.1
        if fs_saturated and op_name == "openbin":
            weights[i] *= 0.5

        # --- Phase-based biasing ---
        if phase == "early":
            if op_name in creators and not (bank_saturated or auth_saturated):
                weights[i] *= 5.0
            if op_name in destructors: weights[i] *= 0.1
        elif phase == "mid":
            if op_name in readers: weights[i] *= 1.7
            if op_name in creators and not (bank_saturated or auth_saturated):
                weights[i] *= 0.8   # reduce — don't spam create once pool reasonable
            if op_name in destructors: weights[i] *= 0.4
        else:  # late
            if op_name in destructors: weights[i] *= 1.5
            if op_name == "advance_clock": weights[i] *= 2.0
            if op_name in creators: weights[i] *= 0.3

    total = sum(weights)
    if total <= 0:
        return rng.choice(ops)
    r = rng.random() * total
    for i, w in enumerate(weights):
        r -= w
        if r <= 0:
            return ops[i]
    return ops[-1]


def gen_trajectory(module, n_steps, rng: random.Random):
    svc = module.new_service()
    traj = {"domain": module.NAME, "source_code": module.SOURCE_CODE, "steps": []}
    for step in range(n_steps):
        phase = _phase_for_step(step, n_steps)
        op_name, arg_names = _pick_op(module, svc, phase, rng)
        args = sample_args(op_name, arg_names, rng, svc, phase=phase, op_name_ctx=op_name)
        method = getattr(svc, op_name)
        try:
            resp = method(**args)
        except TypeError:
            resp = method(*[args[a] for a in arg_names])
        traj["steps"].append({
            "step": step,
            "op": op_name,
            "args": args,
            "response": resp,
            "state_after": svc.state_dict(),
        })
    viols = module.check_invariants(svc.state_dict())
    traj["final_invariant_violations"] = viols
    return traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--domains", nargs="+", default=["bank", "cart", "auth"])
    ap.add_argument("--n-train", type=int, default=500,
                    help="trajectories per domain for training")
    ap.add_argument("--n-eval",  type=int, default=100,
                    help="trajectories per domain for eval (held out)")
    ap.add_argument("--train-steps", type=int, default=80,
                    help="steps per trajectory (training)")
    ap.add_argument("--eval-steps",  type=int, default=200,
                    help="steps per trajectory (eval, LONG-HORIZON)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for dname in args.domains:
        mod = TESTBEDS[dname]
        dom_dir = Path(args.out_dir) / dname
        dom_dir.mkdir(parents=True, exist_ok=True)
        for split, n_traj, n_steps, subseed in [
            ("train", args.n_train, args.train_steps, 0),
            ("eval",  args.n_eval,  args.eval_steps,  10000),
        ]:
            rng = random.Random(args.seed + subseed)
            out = dom_dir / f"{split}.jsonl"
            with out.open("w") as f:
                for i in range(n_traj):
                    t = gen_trajectory(mod, n_steps, random.Random(rng.randrange(2**30)))
                    f.write(json.dumps(t) + "\n")
            print(f"  {dname}/{split}: {n_traj} trajectories x {n_steps} steps -> {out}")

if __name__ == "__main__":
    main()
