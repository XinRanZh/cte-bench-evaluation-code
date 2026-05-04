"""
Testbed 1: Tiny bank. Stateful. Source code IS the ground truth.

The LLM / SCST must learn to simulate this program's behavior from source + call history.
Invariants that can be extracted from source:
  INV-1: balance is always non-negative after any operation
  INV-2: total_deposits - total_withdrawals == sum(balances)  (conservation)
  INV-3: account_ids are unique (no duplicate creation)
  INV-4: a closed account cannot be reopened or transacted on
  INV-5: transfer preserves total assets
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Optional

SOURCE_CODE = '''
class BankService:
    """Stateful bank. All operations atomic. Returns JSON responses."""

    def __init__(self):
        self.accounts: dict[str, float] = {}   # account_id -> balance
        self.closed: set[str] = set()          # closed account ids
        self.total_deposits: float = 0.0
        self.total_withdrawals: float = 0.0

    def create_account(self, account_id: str, initial: float) -> dict:
        if account_id in self.accounts or account_id in self.closed:
            return {"error": {"code": "ACCOUNT_EXISTS", "message": account_id}}
        if initial < 0:
            return {"error": {"code": "NEGATIVE_INITIAL", "message": str(initial)}}
        self.accounts[account_id] = initial
        self.total_deposits += initial
        return {"account_id": account_id, "balance": initial}

    def deposit(self, account_id: str, amount: float) -> dict:
        if account_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": account_id}}
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        if amount <= 0:
            return {"error": {"code": "NON_POSITIVE_AMOUNT", "message": str(amount)}}
        self.accounts[account_id] += amount
        self.total_deposits += amount
        return {"account_id": account_id, "balance": self.accounts[account_id]}

    def withdraw(self, account_id: str, amount: float) -> dict:
        if account_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": account_id}}
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        if amount <= 0:
            return {"error": {"code": "NON_POSITIVE_AMOUNT", "message": str(amount)}}
        if self.accounts[account_id] < amount:
            return {"error": {"code": "INSUFFICIENT_FUNDS",
                              "message": f"balance={self.accounts[account_id]}"}}
        self.accounts[account_id] -= amount
        self.total_withdrawals += amount
        return {"account_id": account_id, "balance": self.accounts[account_id]}

    def transfer(self, from_id: str, to_id: str, amount: float) -> dict:
        if from_id in self.closed or to_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": f"{from_id} or {to_id}"}}
        if from_id not in self.accounts or to_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND",
                              "message": f"{from_id} or {to_id}"}}
        if amount <= 0:
            return {"error": {"code": "NON_POSITIVE_AMOUNT", "message": str(amount)}}
        if self.accounts[from_id] < amount:
            return {"error": {"code": "INSUFFICIENT_FUNDS",
                              "message": f"from_balance={self.accounts[from_id]}"}}
        self.accounts[from_id] -= amount
        self.accounts[to_id]   += amount
        return {"from": {"account_id": from_id, "balance": self.accounts[from_id]},
                "to":   {"account_id": to_id,   "balance": self.accounts[to_id]}}

    def close_account(self, account_id: str) -> dict:
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        bal = self.accounts.pop(account_id)
        self.closed.add(account_id)
        self.total_withdrawals += bal
        return {"account_id": account_id, "closed": True, "final_balance": bal}

    def get_balance(self, account_id: str) -> dict:
        if account_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": account_id}}
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        return {"account_id": account_id, "balance": self.accounts[account_id]}
'''

# -------------------- actual implementation ---------------------------

class BankService:
    """Runtime copy of SOURCE_CODE — kept in sync by hand for development."""

    def __init__(self):
        self.accounts: dict[str, float] = {}
        self.closed: set[str] = set()
        self.total_deposits: float = 0.0
        self.total_withdrawals: float = 0.0

    def create_account(self, account_id, initial):
        if account_id in self.accounts or account_id in self.closed:
            return {"error": {"code": "ACCOUNT_EXISTS", "message": account_id}}
        if initial < 0:
            return {"error": {"code": "NEGATIVE_INITIAL", "message": str(initial)}}
        self.accounts[account_id] = float(initial)
        self.total_deposits += float(initial)
        return {"account_id": account_id, "balance": float(initial)}

    def deposit(self, account_id, amount):
        if account_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": account_id}}
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        if amount <= 0:
            return {"error": {"code": "NON_POSITIVE_AMOUNT", "message": str(amount)}}
        self.accounts[account_id] += amount
        self.total_deposits += amount
        return {"account_id": account_id, "balance": self.accounts[account_id]}

    def withdraw(self, account_id, amount):
        if account_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": account_id}}
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        if amount <= 0:
            return {"error": {"code": "NON_POSITIVE_AMOUNT", "message": str(amount)}}
        if self.accounts[account_id] < amount:
            return {"error": {"code": "INSUFFICIENT_FUNDS",
                              "message": f"balance={self.accounts[account_id]}"}}
        self.accounts[account_id] -= amount
        self.total_withdrawals += amount
        return {"account_id": account_id, "balance": self.accounts[account_id]}

    def transfer(self, from_id, to_id, amount):
        if from_id in self.closed or to_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": f"{from_id} or {to_id}"}}
        if from_id not in self.accounts or to_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": f"{from_id} or {to_id}"}}
        if amount <= 0:
            return {"error": {"code": "NON_POSITIVE_AMOUNT", "message": str(amount)}}
        if self.accounts[from_id] < amount:
            return {"error": {"code": "INSUFFICIENT_FUNDS",
                              "message": f"from_balance={self.accounts[from_id]}"}}
        self.accounts[from_id] -= amount
        self.accounts[to_id]   += amount
        return {"from": {"account_id": from_id, "balance": self.accounts[from_id]},
                "to":   {"account_id": to_id,   "balance": self.accounts[to_id]}}

    def close_account(self, account_id):
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        bal = self.accounts.pop(account_id)
        self.closed.add(account_id)
        self.total_withdrawals += bal
        return {"account_id": account_id, "closed": True, "final_balance": bal}

    def get_balance(self, account_id):
        if account_id in self.closed:
            return {"error": {"code": "ACCOUNT_CLOSED", "message": account_id}}
        if account_id not in self.accounts:
            return {"error": {"code": "ACCOUNT_NOT_FOUND", "message": account_id}}
        return {"account_id": account_id, "balance": self.accounts[account_id]}

    # --- introspection for evaluation only (never shown to simulator) ---
    def state_dict(self):
        return {"accounts": dict(self.accounts), "closed": sorted(self.closed),
                "total_deposits": self.total_deposits,
                "total_withdrawals": self.total_withdrawals}


# -------------------- op grammar -------------------------------------

OPS = [
    ("create_account", ["account_id", "initial"]),
    ("deposit",        ["account_id", "amount"]),
    ("withdraw",       ["account_id", "amount"]),
    ("transfer",       ["from_id", "to_id", "amount"]),
    ("close_account",  ["account_id"]),
    ("get_balance",    ["account_id"]),
]


# -------------------- invariants ------------------------------------

def check_invariants(state: dict) -> list[str]:
    """Return list of violated invariant codes. Operates on state_dict() output."""
    viols = []
    accts = state["accounts"]
    closed = set(state["closed"])
    td, tw = state["total_deposits"], state["total_withdrawals"]

    # INV-1: non-negative balances
    for aid, bal in accts.items():
        if bal < -1e-9:
            viols.append(f"INV1_NEG_BALANCE:{aid}={bal}")

    # INV-2: conservation (approximate; close_account adds remaining to withdrawals)
    total_live = sum(accts.values())
    if abs(td - tw - total_live) > 1e-6:
        viols.append(f"INV2_CONSERVATION:td={td} tw={tw} live={total_live}")

    # INV-3: account id disjointness (accounts keys vs closed set)
    if any(k in closed for k in accts):
        viols.append("INV3_ACTIVE_AND_CLOSED_OVERLAP")

    return viols


# -------------------- helpers for rollout generator ------------------

NAME = "bank"

def new_service():
    return BankService()


# State-probe v2: testbed-level predeclared paths.
PROBE_PATHS_V2 = [
    ["total_deposits"],
    ["total_withdrawals"],
    ("dynamic", "closed_count",
     lambda s: [["__len__", "closed"]]),
    ("dynamic", "account_balances",
     lambda s: [["accounts", a] for a in (s.get("accounts") or {})]),
    ("dynamic", "account_count",
     lambda s: [["__len__", "accounts"]]),
]
