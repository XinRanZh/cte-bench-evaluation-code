"""Extract **simulated-state invariants** from the model's predicted responses
WITHOUT peeking at the hidden ground-truth state.

The invariant contrastive loss needs to know, at training time, which predicted
responses are *consistent with* vs *in violation of* program invariants. For
training, we use the teacher-executed state_after as oracle — but at loss time
we define contrastive pairs over **surface-level invariants** that can be
detected from the response itself.

Invariants we use at training time (surface-visible):
  - A success response must have required keys (e.g., a cart response has
    user_id AND items, or error; never both).
  - A get_balance success balance >= 0.
  - A cart add success: the added product must appear in items.
  - An auth check_access success: granted==true AND user_id matches requested
    as_user_id.

This gives us the positive-set (model outputs that respect invariants) vs
negative-set (that violate) at training time.
"""
from __future__ import annotations
import json
from typing import Any


def check_response_invariants(op: str, args: dict, response) -> list[str]:
    """Return list of violation codes for one (op, args, response) triple.

    Empty list = valid. Non-empty = invariant violation(s) in the response.
    These are SURFACE invariants — checked without hidden state.
    """
    viols = []
    if not isinstance(response, dict):
        return ["RESP_NOT_OBJECT"]
    has_err = "error" in response

    # "Error responses contain ONLY an error key"
    if has_err:
        # Must have a valid code field
        err = response.get("error")
        if not isinstance(err, dict) or "code" not in err:
            viols.append("ERR_SHAPE")
        # Must not contain success-only fields
        for sk in ("balance", "items", "granted", "closed"):
            if sk in response:
                viols.append(f"ERR_HAS_SUCCESS_FIELD:{sk}")
        return viols

    # --- success path per op ---
    if op in ("create_account", "deposit", "withdraw", "get_balance"):
        if "account_id" not in response or "balance" not in response:
            viols.append("BANK_MISSING_FIELDS")
        else:
            if response.get("account_id") != args.get("account_id"):
                viols.append("BANK_ACCT_ID_MISMATCH")
            bal = response["balance"]
            if not isinstance(bal, (int, float)):
                viols.append("BANK_BAL_TYPE")
            elif bal < -1e-9:
                viols.append("BANK_NEG_BALANCE")

    if op == "transfer":
        if not ("from" in response and "to" in response):
            viols.append("TRANSFER_MISSING")
        else:
            fr = response["from"]; to = response["to"]
            for d, key in [(fr, "from_id"), (to, "to_id")]:
                if d.get("account_id") != args.get(key):
                    viols.append("TRANSFER_ID_MISMATCH")
                bal = d.get("balance")
                if not isinstance(bal, (int, float)):
                    viols.append("TRANSFER_BAL_TYPE")
                elif bal < -1e-9:
                    viols.append("TRANSFER_NEG_BAL")

    if op == "close_account":
        if not response.get("closed"):
            viols.append("CLOSE_NOT_TRUE")
        if response.get("account_id") != args.get("account_id"):
            viols.append("CLOSE_ID_MISMATCH")

    if op in ("add_item", "remove_item", "empty_cart", "get_cart"):
        if "user_id" not in response or "items" not in response:
            viols.append("CART_MISSING_FIELDS")
        else:
            if response.get("user_id") != args.get("user_id"):
                viols.append("CART_USER_MISMATCH")
            items = response.get("items")
            if not isinstance(items, list):
                viols.append("CART_ITEMS_NOT_LIST")
            else:
                pids = [it.get("product_id") for it in items if isinstance(it, dict)]
                if len(pids) != len(set(pids)):
                    viols.append("CART_DUP_PID")
                for it in items:
                    if not isinstance(it, dict): continue
                    q = it.get("quantity")
                    if not isinstance(q, (int, float)):
                        viols.append("CART_QTY_TYPE")
                    elif not (1 <= q <= 10):
                        viols.append(f"CART_QTY_RANGE:{q}")
                if op == "add_item":
                    pid = args.get("product_id")
                    if pid not in pids:
                        viols.append("CART_ADDED_ITEM_MISSING")
                if op == "empty_cart":
                    if items != []:
                        viols.append("EMPTY_NOT_EMPTY")

    if op == "issue_token":
        if "token_id" not in response or "user_id" not in response:
            viols.append("TOK_ISSUE_MISSING")
        elif response.get("user_id") != args.get("user_id"):
            viols.append("TOK_ISSUE_USER_MISMATCH")

    if op == "revoke_token":
        if not response.get("revoked"):
            viols.append("REVOKE_NOT_TRUE")

    if op == "check_access":
        if response.get("granted") is not True:
            viols.append("ACCESS_NOT_GRANTED")
        if response.get("user_id") != args.get("as_user_id"):
            viols.append("ACCESS_USER_MISMATCH")

    if op == "advance_clock":
        if "clock" not in response or not isinstance(response["clock"], (int, float)):
            viols.append("CLOCK_MISSING")

    return viols


def is_invariant_violation(op, args, response) -> bool:
    return len(check_response_invariants(op, args, response)) > 0
