"""
Testbed 3: Auth session service — tokens, expiry, revocation.
The trickiest one: long-horizon invariants about token lifecycle.

Invariants:
  INV-1: a token issued to user U, if active, grants access ONLY to U
  INV-2: a token can be revoked at most once
  INV-3: a revoked or expired token cannot grant access
  INV-4: token_id is unique across all history (never reused)
"""
from __future__ import annotations

SOURCE_CODE = '''
class AuthService:
    def __init__(self):
        self.tokens: dict[str, dict] = {}   # tid -> {user_id, ttl, alive}
        self.revoked: set[str] = set()
        self.clock = 0
        self.counter = 0

    def advance_clock(self, delta):
        if delta < 0: return {"error":{"code":"NEGATIVE_DELTA"}}
        self.clock += delta
        for tid, t in list(self.tokens.items()):
            if t["alive"] and t["issued_at"] + t["ttl"] <= self.clock:
                t["alive"] = False
        return {"clock": self.clock}

    def issue_token(self, user_id, ttl):
        if ttl <= 0: return {"error":{"code":"INVALID_TTL"}}
        self.counter += 1
        tid = f"t{self.counter}"
        self.tokens[tid] = {"user_id": user_id, "ttl": ttl,
                            "issued_at": self.clock, "alive": True}
        return {"token_id": tid, "user_id": user_id, "ttl": ttl}

    def revoke_token(self, tid):
        if tid not in self.tokens:
            return {"error":{"code":"TOKEN_NOT_FOUND"}}
        if tid in self.revoked:
            return {"error":{"code":"ALREADY_REVOKED"}}
        self.tokens[tid]["alive"] = False
        self.revoked.add(tid)
        return {"token_id": tid, "revoked": True}

    def check_access(self, tid, as_user_id):
        if tid not in self.tokens:
            return {"error":{"code":"TOKEN_NOT_FOUND"}}
        t = self.tokens[tid]
        if not t["alive"]:
            return {"error":{"code":"TOKEN_INACTIVE"}}
        if t["user_id"] != as_user_id:
            return {"error":{"code":"USER_MISMATCH"}}
        return {"token_id": tid, "granted": True, "user_id": as_user_id}
'''


class AuthService:
    def __init__(self):
        self.tokens = {}
        self.revoked = set()
        self.clock = 0
        self.counter = 0

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        for tid, t in list(self.tokens.items()):
            if t["alive"] and t["issued_at"] + t["ttl"] <= self.clock:
                t["alive"] = False
        return {"clock": self.clock}

    def issue_token(self, user_id, ttl):
        if ttl <= 0:
            return {"error": {"code": "INVALID_TTL"}}
        self.counter += 1
        tid = f"t{self.counter}"
        self.tokens[tid] = {"user_id": user_id, "ttl": ttl,
                            "issued_at": self.clock, "alive": True}
        return {"token_id": tid, "user_id": user_id, "ttl": ttl}

    def revoke_token(self, token_id):
        if token_id not in self.tokens:
            return {"error": {"code": "TOKEN_NOT_FOUND"}}
        if token_id in self.revoked:
            return {"error": {"code": "ALREADY_REVOKED"}}
        self.tokens[token_id]["alive"] = False
        self.revoked.add(token_id)
        return {"token_id": token_id, "revoked": True}

    def check_access(self, token_id, as_user_id):
        if token_id not in self.tokens:
            return {"error": {"code": "TOKEN_NOT_FOUND"}}
        t = self.tokens[token_id]
        if not t["alive"]:
            return {"error": {"code": "TOKEN_INACTIVE"}}
        if t["user_id"] != as_user_id:
            return {"error": {"code": "USER_MISMATCH"}}
        return {"token_id": token_id, "granted": True, "user_id": as_user_id}

    def state_dict(self):
        return {"tokens": {tid: dict(v) for tid, v in self.tokens.items()},
                "revoked": sorted(self.revoked), "clock": self.clock,
                "counter": self.counter}


OPS = [
    ("advance_clock", ["delta"]),
    ("issue_token",   ["user_id", "ttl"]),
    ("revoke_token",  ["token_id"]),
    ("check_access",  ["token_id", "as_user_id"]),
]


def check_invariants(state: dict) -> list[str]:
    viols = []
    toks = state["tokens"]
    revoked = set(state["revoked"])
    tids = list(toks.keys())
    if len(tids) != len(set(tids)):
        viols.append("INV4_DUP_TID")
    for tid in revoked:
        if tid in toks and toks[tid]["alive"]:
            viols.append(f"INV3_REVOKED_ALIVE:{tid}")
    return viols


NAME = "auth"


def new_service():
    return AuthService()


# ---------------------------------------------------------------------------
# State-probe v2: testbed-level predeclared paths.
#
# Each entry is either
#   [path...]                  — a static scalar path
#   ("dynamic", label, fn)     — fn(state) -> list[path], expands at probe time
#
# This removes per-item path declaration (which v1 allowed) and is graded
# with null-baseline subtraction (see scripts/grader.py:grade_state_probe_v2).
PROBE_PATHS_V2 = [
    ["clock"],
    ["counter"],
    ("dynamic", "revoked_count",
     lambda s: [["__len__", "revoked"]]),
    ("dynamic", "token_user_ids",
     lambda s: [["tokens", t, "user_id"] for t in (s.get("tokens") or {})]),
    ("dynamic", "token_alive",
     lambda s: [["tokens", t, "alive"] for t in (s.get("tokens") or {})]),
]
