"""
Testbed 2: Shopping cart (mirrors MIRAGE's Online Boutique cart service).
Stateful, per-user cart with qty limits.

Invariants:
  INV-1: quantity per item in [1, 10]
  INV-2: no duplicate product_id in a single user's cart (merged on add)
  INV-3: empty_cart clears all items; further get_cart returns []
  INV-4: total cart size per user <= 50 items
"""
from __future__ import annotations

SOURCE_CODE = '''
class CartService:
    def __init__(self):
        self.carts: dict[str, list[dict]] = {}  # user_id -> [{product_id, quantity}]

    def add_item(self, user_id, product_id, qty):
        if qty <= 0: return {"error":{"code":"INVALID_QUANTITY","message":str(qty)}}
        if qty > 10: return {"error":{"code":"QUANTITY_EXCEEDS_LIMIT","message":str(qty)}}
        items = self.carts.setdefault(user_id, [])
        for it in items:
            if it["product_id"] == product_id:
                new_q = it["quantity"] + qty
                if new_q > 10:
                    return {"error":{"code":"QUANTITY_EXCEEDS_LIMIT","message":str(new_q)}}
                it["quantity"] = new_q
                return {"user_id":user_id,"items":list(items)}
        if len(items) >= 50:
            return {"error":{"code":"CART_FULL","message":str(len(items))}}
        items.append({"product_id":product_id,"quantity":qty})
        return {"user_id":user_id,"items":list(items)}

    def remove_item(self, user_id, product_id):
        if user_id not in self.carts:
            return {"error":{"code":"CART_NOT_FOUND","message":user_id}}
        items = self.carts[user_id]
        for i, it in enumerate(items):
            if it["product_id"] == product_id:
                items.pop(i)
                return {"user_id":user_id,"items":list(items)}
        return {"error":{"code":"ITEM_NOT_IN_CART","message":product_id}}

    def get_cart(self, user_id):
        return {"user_id":user_id,"items":list(self.carts.get(user_id, []))}

    def empty_cart(self, user_id):
        self.carts[user_id] = []
        return {"user_id":user_id,"items":[]}
'''

class CartService:
    def __init__(self):
        self.carts: dict[str, list[dict]] = {}

    def add_item(self, user_id, product_id, qty):
        if qty <= 0:
            return {"error": {"code": "INVALID_QUANTITY", "message": str(qty)}}
        if qty > 10:
            return {"error": {"code": "QUANTITY_EXCEEDS_LIMIT", "message": str(qty)}}
        items = self.carts.setdefault(user_id, [])
        for it in items:
            if it["product_id"] == product_id:
                new_q = it["quantity"] + qty
                if new_q > 10:
                    return {"error": {"code": "QUANTITY_EXCEEDS_LIMIT", "message": str(new_q)}}
                it["quantity"] = new_q
                return {"user_id": user_id, "items": [dict(x) for x in items]}
        if len(items) >= 50:
            return {"error": {"code": "CART_FULL", "message": str(len(items))}}
        items.append({"product_id": product_id, "quantity": qty})
        return {"user_id": user_id, "items": [dict(x) for x in items]}

    def remove_item(self, user_id, product_id):
        if user_id not in self.carts:
            return {"error": {"code": "CART_NOT_FOUND", "message": user_id}}
        items = self.carts[user_id]
        for i, it in enumerate(items):
            if it["product_id"] == product_id:
                items.pop(i)
                return {"user_id": user_id, "items": [dict(x) for x in items]}
        return {"error": {"code": "ITEM_NOT_IN_CART", "message": product_id}}

    def get_cart(self, user_id):
        return {"user_id": user_id, "items": [dict(x) for x in self.carts.get(user_id, [])]}

    def empty_cart(self, user_id):
        self.carts[user_id] = []
        return {"user_id": user_id, "items": []}

    def state_dict(self):
        return {"carts": {u: [dict(x) for x in its] for u, its in self.carts.items()}}


OPS = [
    ("add_item",    ["user_id", "product_id", "qty"]),
    ("remove_item", ["user_id", "product_id"]),
    ("get_cart",    ["user_id"]),
    ("empty_cart",  ["user_id"]),
]


def check_invariants(state: dict) -> list[str]:
    viols = []
    for uid, items in state["carts"].items():
        pids = [it["product_id"] for it in items]
        if len(pids) != len(set(pids)):
            viols.append(f"INV2_DUP_PID:{uid}")
        if len(items) > 50:
            viols.append(f"INV4_OVER_50:{uid}={len(items)}")
        for it in items:
            q = it["quantity"]
            if not (1 <= q <= 10):
                viols.append(f"INV1_BAD_QTY:{uid}:{it['product_id']}={q}")
    return viols


NAME = "cart"

def new_service():
    return CartService()


# State-probe v2: testbed-level predeclared paths. For each user with a cart
# we probe the number of distinct products and the total quantity.
PROBE_PATHS_V2 = [
    ("dynamic", "cart_size_per_user",
     lambda s: [["__len__", "carts", u] for u in (s.get("carts") or {})]),
    ("dynamic", "total_qty_per_user",
     lambda s: [["__sum_qty__", "carts", u] for u in (s.get("carts") or {})]),
    ("dynamic", "n_users_with_cart",
     lambda s: [["__len__", "carts"]]),
]
