"""
Testbed 4: Reservation service — seats, holds, confirm/cancel, expiry.

Algorithm ported from django-oscar's Basket.add_product / is_quantity_allowed /
flush logic (src/oscar/apps/basket/abstract_models.py, master branch), reduced
to a self-contained module without Django ORM. The guarded threshold
OSCAR_MAX_BASKET_QUANTITY_THRESHOLD is preserved verbatim as MAX_ALLOWED_QUANTITY.

Upstream reference: https://github.com/django-oscar/django-oscar
License: BSD 3-Clause (upstream). This port: research benchmark only.

We add a reservation-lifecycle layer on top of the basket: holds expire on a
logical clock, matching how a real booking platform (events, hotels, airlines)
freezes inventory pending payment.

Invariants:
  INV-1: sum_{h: state in {active, confirmed}} qty[h] + available[s] == capacity[s]
         (confirmed holds continue to consume capacity; expired/cancelled release it)
  INV-2: hold quantities are in [1, MAX_ALLOWED_QUANTITY]
  INV-3: no duplicate (hold_id) across active + expired
  INV-4: a confirmed hold cannot be cancelled; a cancelled hold cannot be confirmed
  INV-5: advance_clock evicts holds whose expires_at <= clock
"""
from __future__ import annotations

MAX_ALLOWED_QUANTITY = 10     # mirrors OSCAR_MAX_BASKET_QUANTITY_THRESHOLD default

SOURCE_CODE = '''
MAX_ALLOWED_QUANTITY = 10

class ReservationService:
    """Seat-inventory service with holds + expiry.
    Ported from django-oscar Basket.add_product quantity/stockrecord checks.
    """

    def __init__(self):
        self.capacity: dict[str, int] = {}       # seat_class_id -> total capacity
        self.available: dict[str, int] = {}      # seat_class_id -> currently free
        self.holds: dict[str, dict] = {}         # hold_id -> {seat, qty, state, expires_at, user_id}
        self.clock: int = 0
        self.counter: int = 0

    def open_class(self, seat_class_id, capacity):
        if seat_class_id in self.capacity:
            return {"error": {"code": "CLASS_EXISTS", "message": seat_class_id}}
        if capacity <= 0:
            return {"error": {"code": "INVALID_CAPACITY", "message": str(capacity)}}
        self.capacity[seat_class_id] = capacity
        self.available[seat_class_id] = capacity
        return {"seat_class_id": seat_class_id, "capacity": capacity}

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        expired = []
        for hid, h in list(self.holds.items()):
            if h["state"] == "active" and h["expires_at"] <= self.clock:
                h["state"] = "expired"
                self.available[h["seat"]] += h["qty"]
                expired.append(hid)
        return {"clock": self.clock, "expired": expired}

    def hold_seats(self, user_id, seat_class_id, qty, ttl):
        # Quantity guard ported verbatim from Oscar is_quantity_allowed:
        # "We enforce a max threshold to prevent a DOS attack via the offers system."
        if qty <= 0:
            return {"error": {"code": "INVALID_QUANTITY", "message": str(qty)}}
        if qty > MAX_ALLOWED_QUANTITY:
            return {"error": {"code": "QUANTITY_EXCEEDS_LIMIT", "message": str(qty)}}
        if seat_class_id not in self.capacity:
            return {"error": {"code": "CLASS_NOT_FOUND", "message": seat_class_id}}
        if ttl <= 0:
            return {"error": {"code": "INVALID_TTL", "message": str(ttl)}}
        if self.available[seat_class_id] < qty:
            return {"error": {"code": "INSUFFICIENT_SEATS",
                              "message": str(self.available[seat_class_id])}}
        self.counter += 1
        hid = f"h{self.counter}"
        self.holds[hid] = {"seat": seat_class_id, "qty": qty, "state": "active",
                           "expires_at": self.clock + ttl, "user_id": user_id}
        self.available[seat_class_id] -= qty
        return {"hold_id": hid, "seat_class_id": seat_class_id, "qty": qty,
                "expires_at": self.clock + ttl}

    def confirm_hold(self, hold_id):
        if hold_id not in self.holds:
            return {"error": {"code": "HOLD_NOT_FOUND", "message": hold_id}}
        h = self.holds[hold_id]
        if h["state"] == "confirmed":
            return {"error": {"code": "ALREADY_CONFIRMED", "message": hold_id}}
        if h["state"] != "active":
            return {"error": {"code": "HOLD_NOT_ACTIVE", "message": h["state"]}}
        h["state"] = "confirmed"
        return {"hold_id": hold_id, "state": "confirmed"}

    def cancel_hold(self, hold_id):
        if hold_id not in self.holds:
            return {"error": {"code": "HOLD_NOT_FOUND", "message": hold_id}}
        h = self.holds[hold_id]
        if h["state"] == "confirmed":
            return {"error": {"code": "CONFIRMED_NOT_CANCELLABLE", "message": hold_id}}
        if h["state"] != "active":
            return {"error": {"code": "HOLD_NOT_ACTIVE", "message": h["state"]}}
        h["state"] = "cancelled"
        self.available[h["seat"]] += h["qty"]
        return {"hold_id": hold_id, "state": "cancelled"}

    def inspect(self, seat_class_id):
        if seat_class_id not in self.capacity:
            return {"error": {"code": "CLASS_NOT_FOUND", "message": seat_class_id}}
        return {"seat_class_id": seat_class_id,
                "capacity": self.capacity[seat_class_id],
                "available": self.available[seat_class_id]}
'''


class ReservationService:
    def __init__(self):
        self.capacity: dict[str, int] = {}
        self.available: dict[str, int] = {}
        self.holds: dict[str, dict] = {}
        self.clock: int = 0
        self.counter: int = 0

    def open_class(self, seat_class_id, capacity):
        if seat_class_id in self.capacity:
            return {"error": {"code": "CLASS_EXISTS", "message": seat_class_id}}
        if capacity <= 0:
            return {"error": {"code": "INVALID_CAPACITY", "message": str(capacity)}}
        self.capacity[seat_class_id] = capacity
        self.available[seat_class_id] = capacity
        return {"seat_class_id": seat_class_id, "capacity": capacity}

    def advance_clock(self, delta):
        if delta < 0:
            return {"error": {"code": "NEGATIVE_DELTA"}}
        self.clock += delta
        expired = []
        for hid, h in list(self.holds.items()):
            if h["state"] == "active" and h["expires_at"] <= self.clock:
                h["state"] = "expired"
                self.available[h["seat"]] += h["qty"]
                expired.append(hid)
        return {"clock": self.clock, "expired": expired}

    def hold_seats(self, user_id, seat_class_id, qty, ttl):
        if qty <= 0:
            return {"error": {"code": "INVALID_QUANTITY", "message": str(qty)}}
        if qty > MAX_ALLOWED_QUANTITY:
            return {"error": {"code": "QUANTITY_EXCEEDS_LIMIT", "message": str(qty)}}
        if seat_class_id not in self.capacity:
            return {"error": {"code": "CLASS_NOT_FOUND", "message": seat_class_id}}
        if ttl <= 0:
            return {"error": {"code": "INVALID_TTL", "message": str(ttl)}}
        if self.available[seat_class_id] < qty:
            return {"error": {"code": "INSUFFICIENT_SEATS",
                              "message": str(self.available[seat_class_id])}}
        self.counter += 1
        hid = f"h{self.counter}"
        self.holds[hid] = {"seat": seat_class_id, "qty": qty, "state": "active",
                           "expires_at": self.clock + ttl, "user_id": user_id}
        self.available[seat_class_id] -= qty
        return {"hold_id": hid, "seat_class_id": seat_class_id, "qty": qty,
                "expires_at": self.clock + ttl}

    def confirm_hold(self, hold_id):
        if hold_id not in self.holds:
            return {"error": {"code": "HOLD_NOT_FOUND", "message": hold_id}}
        h = self.holds[hold_id]
        if h["state"] == "confirmed":
            return {"error": {"code": "ALREADY_CONFIRMED", "message": hold_id}}
        if h["state"] != "active":
            return {"error": {"code": "HOLD_NOT_ACTIVE", "message": h["state"]}}
        h["state"] = "confirmed"
        return {"hold_id": hold_id, "state": "confirmed"}

    def cancel_hold(self, hold_id):
        if hold_id not in self.holds:
            return {"error": {"code": "HOLD_NOT_FOUND", "message": hold_id}}
        h = self.holds[hold_id]
        if h["state"] == "confirmed":
            return {"error": {"code": "CONFIRMED_NOT_CANCELLABLE", "message": hold_id}}
        if h["state"] != "active":
            return {"error": {"code": "HOLD_NOT_ACTIVE", "message": h["state"]}}
        h["state"] = "cancelled"
        self.available[h["seat"]] += h["qty"]
        return {"hold_id": hold_id, "state": "cancelled"}

    def inspect(self, seat_class_id):
        if seat_class_id not in self.capacity:
            return {"error": {"code": "CLASS_NOT_FOUND", "message": seat_class_id}}
        return {"seat_class_id": seat_class_id,
                "capacity": self.capacity[seat_class_id],
                "available": self.available[seat_class_id]}

    def state_dict(self):
        return {"capacity": dict(self.capacity),
                "available": dict(self.available),
                "holds": {hid: dict(h) for hid, h in self.holds.items()},
                "clock": self.clock, "counter": self.counter}


OPS = [
    ("open_class",    ["seat_class_id", "capacity"]),
    ("advance_clock", ["delta"]),
    ("hold_seats",    ["user_id", "seat_class_id", "qty", "ttl"]),
    ("confirm_hold",  ["hold_id"]),
    ("cancel_hold",   ["hold_id"]),
    ("inspect",       ["seat_class_id"]),
]


def check_invariants(state: dict) -> list[str]:
    viols = []
    caps = state["capacity"]
    avail = state["available"]
    holds = state["holds"]
    hids = list(holds.keys())
    if len(hids) != len(set(hids)):
        viols.append("INV3_DUP_HID")
    # INV-1: conservation per seat class. Active and confirmed holds consume
    # capacity; expired and cancelled holds have already returned seats.
    held_by_class: dict[str, int] = {s: 0 for s in caps}
    for hid, h in holds.items():
        if h["state"] in ("active", "confirmed"):
            held_by_class[h["seat"]] = held_by_class.get(h["seat"], 0) + h["qty"]
        if not (1 <= h["qty"] <= MAX_ALLOWED_QUANTITY):
            viols.append(f"INV2_BAD_QTY:{hid}={h['qty']}")
    for s, cap in caps.items():
        if held_by_class.get(s, 0) + avail.get(s, 0) != cap:
            viols.append(f"INV1_CONSERVATION:{s}:"
                         f"held={held_by_class.get(s, 0)}+avail={avail.get(s, 0)}!=cap={cap}")
    return viols


NAME = "reservation"


def new_service():
    return ReservationService()


# State-probe v2: testbed-level predeclared paths.
PROBE_PATHS_V2 = [
    ["clock"],
    ["counter"],
    ("dynamic", "capacity_per_class",
     lambda s: [["capacity", c] for c in (s.get("capacity") or {})]),
    ("dynamic", "available_per_class",
     lambda s: [["available", c] for c in (s.get("available") or {})]),
    ("dynamic", "hold_count",
     lambda s: [["__len__", "holds"]]),
    ("dynamic", "hold_qty",
     lambda s: [["holds", h, "qty"] for h in (s.get("holds") or {})]),
]
