"""
Testbed 5: Rate limiter — per-key token bucket + sliding-window.

Algorithm ported verbatim from python-limits (alisaifee/limits, master branch):
limits/storage/memory.py incr / get / acquire_sliding_window_entry, and
limits/strategies.py fixed-window semantics, adapted to a logical clock.

Upstream reference: https://github.com/alisaifee/limits
License: MIT (upstream). This port: research benchmark only.

The wall-clock time.time() in upstream is replaced by an explicit logical clock
so that counterfactual do(clock <- t') is well-defined. Sliding-window weighting
preserves upstream's previous_count * previous_ttl / expiry + current_count.

Invariants:
  INV-1: fixed-window counter for (key, window) never exceeds LIMIT on accept=True
  INV-2: weighted sliding count <= LIMIT on accept=True
  INV-3: reset() clears all counters
  INV-4: expired counters are evicted on next access (lazy expiry, upstream semantics)
"""
from __future__ import annotations
from math import floor

SOURCE_CODE = '''
from math import floor

class RateLimiter:
    """Fixed-window and sliding-window counter limiter with logical clock.
    Ported from limits.storage.memory.MemoryStorage.
    """

    def __init__(self, limit, expiry):
        # limit: max events per window; expiry: window length in logical ticks.
        self.limit = limit
        self.expiry = expiry
        self.storage: dict[str, int] = {}     # key -> count
        self.expirations: dict[str, int] = {} # key -> tick at which counter expires
        self.clock = 0

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        return {"clock": self.clock}

    def _get(self, key):
        # lazy expiry, upstream:
        # if self.expirations.get(key, 0) <= time.time(): pop.
        if self.expirations.get(key, 0) <= self.clock:
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
        return self.storage.get(key, 0)

    def _incr(self, key, expiry, amount=1):
        self._get(key)
        self.storage[key] = self.storage.get(key, 0) + amount
        if self.storage[key] == amount:
            self.expirations[key] = self.clock + expiry
        return self.storage[key]

    def _window_start(self):
        # fixed window anchored to floor(clock/expiry)*expiry.
        return (self.clock // self.expiry) * self.expiry

    def hit_fixed(self, key, amount=1):
        if amount <= 0:
            return {"error": {"code": "INVALID_AMOUNT", "message": str(amount)}}
        if amount > self.limit:
            return {"accepted": False, "key": key, "count": 0, "reason": "amount>limit"}
        window_key = f"{key}:{self._window_start()}"
        current = self._get(window_key)
        if current + amount > self.limit:
            return {"accepted": False, "key": key, "count": current,
                    "reason": "would_exceed_limit"}
        new_count = self._incr(window_key, self.expiry, amount)
        return {"accepted": True, "key": key, "count": new_count}

    def hit_sliding(self, key, amount=1):
        # Ported from MemoryStorage.acquire_sliding_window_entry.
        if amount <= 0:
            return {"error": {"code": "INVALID_AMOUNT", "message": str(amount)}}
        if amount > self.limit:
            return {"accepted": False, "key": key, "weighted": 0.0,
                    "reason": "amount>limit"}
        previous_key = f"{key}:prev:{(self.clock - self.expiry) // self.expiry}"
        current_key  = f"{key}:curr:{self.clock // self.expiry}"
        previous_count = self._get(previous_key)
        current_count  = self._get(current_key)
        if previous_count == 0:
            previous_ttl = 0.0
        else:
            previous_ttl = (1 - (((self.clock - self.expiry) / self.expiry) % 1)) * self.expiry
        weighted = previous_count * previous_ttl / self.expiry + current_count
        if floor(weighted) + amount > self.limit:
            return {"accepted": False, "key": key, "weighted": weighted,
                    "reason": "would_exceed_limit"}
        new_current = self._incr(current_key, 2 * self.expiry, amount)
        weighted = previous_count * previous_ttl / self.expiry + new_current
        if floor(weighted) > self.limit:
            # upstream rollback branch.
            self.storage[current_key] -= amount
            return {"accepted": False, "key": key, "weighted": weighted,
                    "reason": "rollback"}
        return {"accepted": True, "key": key, "weighted": weighted}

    def get_count(self, key, strategy):
        if strategy == "fixed":
            window_key = f"{key}:{self._window_start()}"
            return {"key": key, "count": self._get(window_key),
                    "strategy": "fixed"}
        if strategy == "sliding":
            current_key = f"{key}:curr:{self.clock // self.expiry}"
            return {"key": key, "count": self._get(current_key),
                    "strategy": "sliding"}
        return {"error": {"code": "UNKNOWN_STRATEGY", "message": strategy}}

    def reset(self):
        n = len(self.storage)
        self.storage.clear()
        self.expirations.clear()
        return {"cleared": n}
'''


class RateLimiter:
    def __init__(self, limit: int = 5, expiry: int = 10):
        self.limit = limit
        self.expiry = expiry
        self.storage: dict[str, int] = {}
        self.expirations: dict[str, int] = {}
        self.clock: int = 0

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        return {"clock": self.clock}

    def _get(self, key):
        if self.expirations.get(key, 0) <= self.clock:
            self.storage.pop(key, None)
            self.expirations.pop(key, None)
        return self.storage.get(key, 0)

    def _incr(self, key, expiry, amount=1):
        self._get(key)
        self.storage[key] = self.storage.get(key, 0) + amount
        if self.storage[key] == amount:
            self.expirations[key] = self.clock + expiry
        return self.storage[key]

    def _window_start(self):
        return (self.clock // self.expiry) * self.expiry

    def hit_fixed(self, key, amount=1):
        if amount <= 0:
            return {"error": {"code": "INVALID_AMOUNT", "message": str(amount)}}
        if amount > self.limit:
            return {"accepted": False, "key": key, "count": 0, "reason": "amount>limit"}
        window_key = f"{key}:{self._window_start()}"
        current = self._get(window_key)
        if current + amount > self.limit:
            return {"accepted": False, "key": key, "count": current,
                    "reason": "would_exceed_limit"}
        new_count = self._incr(window_key, self.expiry, amount)
        return {"accepted": True, "key": key, "count": new_count}

    def hit_sliding(self, key, amount=1):
        if amount <= 0:
            return {"error": {"code": "INVALID_AMOUNT", "message": str(amount)}}
        if amount > self.limit:
            return {"accepted": False, "key": key, "weighted": 0.0,
                    "reason": "amount>limit"}
        previous_key = f"{key}:prev:{(self.clock - self.expiry) // self.expiry}"
        current_key  = f"{key}:curr:{self.clock // self.expiry}"
        previous_count = self._get(previous_key)
        current_count  = self._get(current_key)
        if previous_count == 0:
            previous_ttl = 0.0
        else:
            previous_ttl = (1 - (((self.clock - self.expiry) / self.expiry) % 1)) * self.expiry
        weighted = previous_count * previous_ttl / self.expiry + current_count
        if floor(weighted) + amount > self.limit:
            return {"accepted": False, "key": key, "weighted": weighted,
                    "reason": "would_exceed_limit"}
        new_current = self._incr(current_key, 2 * self.expiry, amount)
        weighted = previous_count * previous_ttl / self.expiry + new_current
        if floor(weighted) > self.limit:
            self.storage[current_key] -= amount
            return {"accepted": False, "key": key, "weighted": weighted,
                    "reason": "rollback"}
        return {"accepted": True, "key": key, "weighted": weighted}

    def get_count(self, key, strategy):
        if strategy == "fixed":
            window_key = f"{key}:{self._window_start()}"
            return {"key": key, "count": self._get(window_key), "strategy": "fixed"}
        if strategy == "sliding":
            current_key = f"{key}:curr:{self.clock // self.expiry}"
            return {"key": key, "count": self._get(current_key), "strategy": "sliding"}
        return {"error": {"code": "UNKNOWN_STRATEGY", "message": strategy}}

    def reset(self):
        n = len(self.storage)
        self.storage.clear()
        self.expirations.clear()
        return {"cleared": n}

    def state_dict(self):
        return {"limit": self.limit, "expiry": self.expiry,
                "storage": dict(self.storage),
                "expirations": dict(self.expirations),
                "clock": self.clock}


OPS = [
    ("advance_clock", ["delta"]),
    ("hit_fixed",     ["key", "amount"]),
    ("hit_sliding",   ["key", "amount"]),
    ("get_count",     ["key", "strategy"]),
    ("reset",         []),
]


def check_invariants(state: dict) -> list[str]:
    viols = []
    limit = state["limit"]
    # INV-1: no counter exceeds LIMIT.
    for key, count in state["storage"].items():
        if count > limit:
            viols.append(f"INV1_COUNT_OVER_LIMIT:{key}={count}>{limit}")
    # INV-4: no counter is retained past its expiration + current clock.
    clock = state["clock"]
    for key, exp in state["expirations"].items():
        if exp <= clock and key in state["storage"]:
            viols.append(f"INV4_LIVE_EXPIRED:{key}:exp={exp}<=clock={clock}")
    return viols


NAME = "ratelimiter"


def new_service():
    return RateLimiter()


# State-probe v2: testbed-level predeclared paths.
PROBE_PATHS_V2 = [
    ["clock"],
    ["limit"],
    ["expiry"],
    ("dynamic", "counter_per_key",
     lambda s: [["storage", k] for k in (s.get("storage") or {})]),
    ("dynamic", "expiration_per_key",
     lambda s: [["expirations", k] for k in (s.get("expirations") or {})]),
    ("dynamic", "n_active_keys",
     lambda s: [["__len__", "storage"]]),
]
