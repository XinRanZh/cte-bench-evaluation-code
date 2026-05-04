"""Adversarial hard-subset patches: delayed-effect, non-local source edits.

Each entry is a `CodeInterventionSpec`-compatible record whose one-line
source mutation produces responses that are IDENTICAL to the original for
the first `expected_delay` steps in a typical trajectory, then diverge
sharply. This is the subset designed to break the 94% value-match ceiling
observed on Kimi K2.5 / full-history.

Construction principles (per GPT-5.5 xhigh recommendation, 2026-04-29):

  1. The patch must be *syntactically local*: one comparator or one numeric
     literal. Explanation-localization remains gradeable.
  2. The response shape of the patched trace must be IDENTICAL for at least
     ~N steps in the empirical trajectory distribution, where N is tuned per
     domain to be ≥ 10.
  3. After the delay, the divergence must be NONLOCAL: it should cascade
     into the visible response of later operations, not only into
     `state_dict()`. Otherwise the value-match metric would be blind.
  4. Invariants of the UNPATCHED service must continue to hold on the
     patched state — an adversarial patch is not an invariant-violation
     generator. We're testing whether the LLM can tell two invariant-
     preserving programs apart, not whether it catches bugs.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AdversarialPatch:
    name: str
    domain: str
    description: str
    # Exact find/replace on the SOURCE_CODE string. If `anchor` is not None,
    # we require the anchor appears in the source (for version safety) and
    # we replace only the first occurrence of `find` *after* the anchor.
    anchor: str | None
    find: str
    replace: str
    expected_delay: int
    rationale: str


PATCHES: list[AdversarialPatch] = [
    # ---- auth: off-by-one on TTL expiry predicate ----
    AdversarialPatch(
        name="auth_ttl_off_by_one",
        domain="auth",
        description="TTL expiry predicate `issued_at + ttl <= clock` "
                    "becomes `< clock`. Tokens stay alive exactly one tick "
                    "longer than ground truth.",
        anchor="def advance_clock",
        find="t[\"issued_at\"] + t[\"ttl\"] <= self.clock:",
        replace="t[\"issued_at\"] + t[\"ttl\"] <  self.clock:",
        expected_delay=15,
        rationale=("invisible until a `check_access` follows `advance_clock` "
                   "by EXACTLY the ttl; model must track the boundary off-by-one."),
    ),

    # ---- reservation: off-by-one on hold expiry ----
    AdversarialPatch(
        name="reservation_expiry_off_by_one",
        domain="reservation",
        description="Hold expiry predicate `expires_at <= clock` becomes "
                    "`expires_at < clock`. A hold with expires_at==clock "
                    "stays active for one more tick.",
        anchor="def advance_clock",
        find="h[\"expires_at\"] <= self.clock:",
        replace="h[\"expires_at\"] <  self.clock:",
        expected_delay=12,
        rationale=("invisible until the first exact-boundary advance_clock; "
                   "then downstream `available` and `inspect` diverge."),
    ),

    # ---- ratelimiter: fixed-window drift by 1 tick ----
    AdversarialPatch(
        name="ratelimiter_window_drift",
        domain="ratelimiter",
        description="Fixed-window anchor shifts by 1 tick: "
                    "`(self.clock // self.expiry) * self.expiry` becomes "
                    "`((self.clock - 1) // self.expiry) * self.expiry`. The "
                    "first window boundary is hit one tick earlier.",
        anchor="def _window_start",
        find="return (self.clock // self.expiry) * self.expiry",
        replace="return ((self.clock - 1) // self.expiry) * self.expiry",
        expected_delay=10,
        rationale=("invisible until the trajectory actually crosses a window "
                   "boundary; then `hit_fixed` accept/reject flips on the boundary tick."),
    ),

    # ---- filesystem: lock-expiry off-by-one ----
    AdversarialPatch(
        name="filesystem_lock_expiry_off_by_one",
        domain="filesystem",
        description="Lock expiry predicate `hold_until <= clock` becomes "
                    "`hold_until < clock`. Locks persist one extra tick.",
        anchor="def advance_clock",
        find="if l[\"hold_until\"] <= self.clock:",
        replace="if l[\"hold_until\"] <  self.clock:",
        expected_delay=18,
        rationale=("bites only when a write_bytes or lock follows "
                   "advance_clock by exactly the lock's ttl."),
    ),

    # ---- cart: qty=7 bypass (delayed non-local) ----
    AdversarialPatch(
        name="cart_qty7_bypass",
        domain="cart",
        description="`qty > 10` guard becomes `qty > 10 and qty != 7`. "
                    "Add_item with qty=7 bypasses the limit; first visible "
                    "effect is that a cart can exceed total=10 when qty=7 "
                    "is added twice.",
        anchor="def add_item",
        find="if qty > 10:",
        replace="if qty > 10 and qty != 7:",
        expected_delay=20,
        rationale=("most sample_args draws qty uniformly in [1,10]; qty=7 "
                   "appears ~every 10 add_item calls, and the divergence "
                   "only manifests when the same user adds qty=7 twice."),
    ),

    # ---- bank: silent accumulator sign ----
    AdversarialPatch(
        name="bank_withdrawal_sign",
        domain="bank",
        description="`self.total_withdrawals += amount` becomes `-= amount`. "
                    "Per-account balance is unaffected (primary value-match "
                    "remains correct), but `state_dict().total_withdrawals` "
                    "diverges immediately on first withdraw. Detected by "
                    "explanation-localization + invariant checker running on "
                    "the LLM's predicted state.",
        anchor="def withdraw",
        find="self.total_withdrawals += amount",
        replace="self.total_withdrawals -= amount",
        expected_delay=5,
        rationale=("value-match on `balance` is unaffected — we're testing "
                   "whether LLMs track non-primary state variables."),
    ),
]


def get_patches(domain: str | None = None) -> list[AdversarialPatch]:
    if domain is None:
        return list(PATCHES)
    return [p for p in PATCHES if p.domain == domain]


def apply_adversarial_patch(source: str, patch: AdversarialPatch
                            ) -> tuple[str, dict]:
    """Apply a named adversarial patch to SOURCE_CODE.

    Returns (patched_source, spec_dict). Raises if the anchor or find string
    does not appear exactly once in the applicable region — we never want to
    silently mispatch.
    """
    if patch.anchor is not None:
        anchor_idx = source.find(patch.anchor)
        if anchor_idx == -1:
            raise RuntimeError(
                f"adversarial patch {patch.name}: anchor {patch.anchor!r} "
                f"not found in {patch.domain} source")
        search_region_start = anchor_idx
    else:
        search_region_start = 0

    region = source[search_region_start:]
    occurrences = region.count(patch.find)
    if occurrences == 0:
        raise RuntimeError(
            f"adversarial patch {patch.name}: find {patch.find!r} not found "
            f"after anchor in {patch.domain} source")
    if occurrences > 1:
        raise RuntimeError(
            f"adversarial patch {patch.name}: find {patch.find!r} ambiguous "
            f"({occurrences} occurrences) in {patch.domain} source")

    idx = search_region_start + region.find(patch.find)
    patched = source[:idx] + patch.replace + source[idx + len(patch.find):]

    # Compute 1-indexed line number of the change for explanation-localization.
    line_number = source[:idx].count("\n") + 1
    # Column = idx - (start of that line)
    line_start = source.rfind("\n", 0, idx) + 1
    col_offset = idx - line_start

    spec = {
        "kind": "adversarial",
        "name": patch.name,
        "domain": patch.domain,
        "description": patch.description,
        "line_number": line_number,
        "col_offset": col_offset,
        "original_span": patch.find,
        "modified_span": patch.replace,
        "expected_delay": patch.expected_delay,
        "rationale": patch.rationale,
    }
    return patched, spec
