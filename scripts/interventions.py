"""Counterfactual intervention primitives.

Each primitive takes a `SOURCE_CODE` string (the same one the LLM sees) and
returns a mutated source plus an `explanation_truth` record with the exact
1-indexed line number, original snippet, and modified snippet — that is the
ground truth for the explanation-localization metric.

We use AST to *locate* the change and text-level substitution (splitlines +
column offset) to *apply* it. This keeps line numbers stable between the
original and patched source, which matters for deterministic grading.

Five intervention families:

  (a) threshold_change    — a numeric literal in a comparison (e.g. `qty > 10`
                            → `qty > 5`).
  (b) branch_inversion    — a comparator or boolean operator is inverted (e.g.
                            `<=` → `<`; `if x:` → `if not x:`).
  (c) quota_change        — a module-level constant assignment (e.g.
                            `MAX_ALLOWED_QUANTITY = 10` → `= 5`).
  (d) clock_jump          — not a source patch, a *state* intervention that
                            jumps a `clock` field forward (handled via
                            `state_injection` with `target.path="clock"`).
  (e) state_injection     — a scalar / container field in `state_dict()` is
                            overwritten at step k; realized by the
                            orchestrator, not here.

This module implements (a) (b) (c). State-level interventions (d), (e) are
expressed as `StateInterventionSpec` dataclasses and applied by the
orchestrator; we provide the dataclass and a serializer here for symmetry.
"""
from __future__ import annotations
import ast
import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Intervention specs

@dataclass
class CodeInterventionSpec:
    kind: str                    # "threshold_change" | "branch_inversion" | "quota_change"
    domain: str                  # "bank" | "cart" | "auth" | "reservation" | "ratelimiter" | "filesystem"
    line_number: int             # 1-indexed line in SOURCE_CODE
    col_offset: int              # 0-indexed column start of the replaced span
    original_span: str           # exact text replaced
    modified_span: str           # exact text used as replacement
    description: str             # human-readable


@dataclass
class StateInterventionSpec:
    kind: str                    # "state_injection" | "clock_jump"
    domain: str
    path: list                   # e.g. ["available", "A"] or ["tokens", "t3", "alive"]
    from_value: object
    to_value: object
    description: str


# ---------------------------------------------------------------------------
# AST helpers

def _iter_comparisons(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for i, (op, rhs) in enumerate(zip(node.ops, node.comparators)):
                lhs = node.left if i == 0 else node.comparators[i - 1]
                yield node, lhs, op, rhs


def _iter_module_constants(tree: ast.AST):
    for node in tree.body:
        if (isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)):
            yield node, node.targets[0].id, node.value


# ---------------------------------------------------------------------------
# Text-level patcher that preserves line numbers.

def _apply_span_replacement(source: str, lineno: int, col: int,
                            original: str, modified: str) -> str:
    """Replace `original` at (1-indexed lineno, 0-indexed col) with `modified`.

    Preserves trailing newline structure. Raises if the slice doesn't match
    `original` exactly (defensive: we never silently mispatch).
    """
    lines = source.splitlines(keepends=True)
    assert 1 <= lineno <= len(lines), (lineno, len(lines))
    line = lines[lineno - 1]
    end = col + len(original)
    found = line[col:end]
    if found != original:
        raise RuntimeError(
            f"Span mismatch at line {lineno} col {col}: "
            f"expected {original!r}, found {found!r} in line {line!r}")
    lines[lineno - 1] = line[:col] + modified + line[end:]
    return "".join(lines)


# ---------------------------------------------------------------------------
# Primitive: threshold_change

def find_threshold_candidates(source: str) -> list[tuple[int, int, str, str]]:
    """Return candidate (lineno, col, original_token, suggested_replacement).

    A candidate is a numeric-constant RHS of a comparison where the operator
    is one of {<, <=, >, >=, ==}. We suggest a non-trivial perturbation (half
    for >=2, doubled for small ints, off-by-one otherwise).
    """
    tree = ast.parse(source)
    out = []
    for _, lhs, op, rhs in _iter_comparisons(tree):
        if not (isinstance(rhs, ast.Constant) and isinstance(rhs.value, (int, float))):
            continue
        if not isinstance(op, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq)):
            continue
        v = rhs.value
        if isinstance(v, float):
            alt = v / 2 if v >= 2 else v + 1
        else:
            if v >= 2:     alt = v // 2
            elif v == 1:   alt = 0
            elif v == 0:   alt = 1
            else:          alt = v + 1
        original = repr(v) if isinstance(v, float) else str(v)
        modified = repr(alt) if isinstance(alt, float) else str(alt)
        if original == modified:
            continue
        out.append((rhs.lineno, rhs.col_offset, original, modified))
    return out


def apply_threshold_change(source: str, domain: str,
                           choice_index: int = 0,
                           lineno: Optional[int] = None) -> tuple[str, CodeInterventionSpec]:
    cands = find_threshold_candidates(source)
    if not cands:
        raise ValueError(f"no threshold candidates in {domain}")
    if lineno is not None:
        cands = [c for c in cands if c[0] == lineno] or cands
    ln, col, orig, mod = cands[choice_index % len(cands)]
    new_src = _apply_span_replacement(source, ln, col, orig, mod)
    spec = CodeInterventionSpec(
        kind="threshold_change", domain=domain,
        line_number=ln, col_offset=col,
        original_span=orig, modified_span=mod,
        description=f"numeric threshold on line {ln} lowered from {orig} to {mod}")
    return new_src, spec


# ---------------------------------------------------------------------------
# Primitive: branch_inversion

_COMPARATOR_FLIP = {
    "<=": ">",   ">=": "<",   "<": ">=",   ">": "<=",
    "==": "!=",  "!=": "==",
}


def find_branch_candidates(source: str) -> list[tuple[int, int, str, str]]:
    """Locate comparator tokens suitable for inversion.

    Approach: parse AST, find each `ast.Compare` and for each op_node locate
    its text span by looking in the line at lhs.end_col_offset .. rhs.col_offset
    and matching the operator token.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    out = []
    for _, lhs, op, rhs in _iter_comparisons(tree):
        op_symbol = {
            ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=",
            ast.Eq: "==", ast.NotEq: "!=",
        }.get(type(op))
        if op_symbol is None:
            continue
        # Locate the operator in the lhs line between lhs end and rhs start
        if not (hasattr(lhs, "end_lineno") and hasattr(lhs, "end_col_offset")):
            continue
        # only handle same-line comparisons (rare in our testbeds to span lines)
        if lhs.end_lineno != rhs.lineno:
            continue
        line = lines[lhs.end_lineno - 1]
        search_start = lhs.end_col_offset
        search_end = rhs.col_offset
        span_text = line[search_start:search_end]
        # find op_symbol in span_text; prefer exact match of the full token
        # (handles "<=" before "<" because we look up flip dict keyed by full).
        for token in ("<=", ">=", "==", "!=", "<", ">"):
            if token != op_symbol:
                continue
            idx = span_text.find(token)
            if idx == -1:
                continue
            col = search_start + idx
            out.append((lhs.end_lineno, col, token, _COMPARATOR_FLIP[token]))
            break
    return out


def apply_branch_inversion(source: str, domain: str,
                           choice_index: int = 0,
                           lineno: Optional[int] = None) -> tuple[str, CodeInterventionSpec]:
    cands = find_branch_candidates(source)
    if not cands:
        raise ValueError(f"no branch-inversion candidates in {domain}")
    if lineno is not None:
        cands = [c for c in cands if c[0] == lineno] or cands
    ln, col, orig, mod = cands[choice_index % len(cands)]
    new_src = _apply_span_replacement(source, ln, col, orig, mod)
    spec = CodeInterventionSpec(
        kind="branch_inversion", domain=domain,
        line_number=ln, col_offset=col,
        original_span=orig, modified_span=mod,
        description=f"comparator on line {ln} inverted from `{orig}` to `{mod}`")
    return new_src, spec


# ---------------------------------------------------------------------------
# Primitive: quota_change (module-level constant assignment)

def find_quota_candidates(source: str) -> list[tuple[int, int, str, str, str]]:
    """Return (lineno, col, original_literal, suggested_literal, name)."""
    tree = ast.parse(source)
    out = []
    for assign, name, const_node in _iter_module_constants(tree):
        v = const_node.value
        if not isinstance(v, (int, float)):
            continue
        if isinstance(v, float):
            alt = v / 2 if v >= 2 else v + 1
        else:
            if v >= 2:     alt = v // 2
            elif v == 1:   alt = 100
            elif v == 0:   alt = 1
            else:          alt = v + 1
        if alt == v:
            continue
        original = repr(v) if isinstance(v, float) else str(v)
        modified = repr(alt) if isinstance(alt, float) else str(alt)
        out.append((const_node.lineno, const_node.col_offset, original, modified, name))
    return out


def apply_quota_change(source: str, domain: str,
                      choice_index: int = 0,
                      lineno: Optional[int] = None) -> tuple[str, CodeInterventionSpec]:
    cands = find_quota_candidates(source)
    if not cands:
        raise ValueError(f"no quota candidates in {domain}")
    if lineno is not None:
        cands = [c for c in cands if c[0] == lineno] or cands
    ln, col, orig, mod, name = cands[choice_index % len(cands)]
    new_src = _apply_span_replacement(source, ln, col, orig, mod)
    spec = CodeInterventionSpec(
        kind="quota_change", domain=domain,
        line_number=ln, col_offset=col,
        original_span=orig, modified_span=mod,
        description=f"module constant {name} (line {ln}) changed from {orig} to {mod}")
    return new_src, spec


# ---------------------------------------------------------------------------
# Dispatcher

CODE_INTERVENTIONS = {
    "threshold_change": apply_threshold_change,
    "branch_inversion": apply_branch_inversion,
    "quota_change":     apply_quota_change,
}


def apply_code_intervention(kind: str, source: str, domain: str,
                            **kwargs) -> tuple[str, CodeInterventionSpec]:
    if kind not in CODE_INTERVENTIONS:
        raise ValueError(f"unknown code intervention kind: {kind}")
    return CODE_INTERVENTIONS[kind](source, domain, **kwargs)


# ---------------------------------------------------------------------------
# State interventions (applied by the orchestrator after reachability filter)

def make_state_injection(domain: str, path: list,
                         from_value, to_value,
                         kind: str = "state_injection") -> StateInterventionSpec:
    return StateInterventionSpec(
        kind=kind, domain=domain, path=list(path),
        from_value=from_value, to_value=to_value,
        description=f"state[{'.'.join(map(str, path))}] overwritten "
                    f"from {from_value!r} to {to_value!r}")


def apply_state_mutation(state: dict, path: list, to_value) -> dict:
    """Return a new state dict with `state[*path] = to_value`. Non-destructive.

    Supports dict[str]->dict[str]->... and dict[str]->list[int] paths. Raises
    KeyError/IndexError if the path does not exist (we don't create new keys —
    interventions are local edits, not structural rewrites).
    """
    if not path:
        raise ValueError("empty path")
    import copy
    new_state = copy.deepcopy(state)
    cur = new_state
    for step in path[:-1]:
        if isinstance(cur, dict):
            cur = cur[step]
        elif isinstance(cur, list):
            cur = cur[int(step)]
        else:
            raise TypeError(f"cannot descend into {type(cur).__name__} at step {step!r}")
    last = path[-1]
    if isinstance(cur, dict):
        if last not in cur:
            raise KeyError(f"path {path} missing at {last!r}")
        cur[last] = to_value
    elif isinstance(cur, list):
        cur[int(last)] = to_value
    else:
        raise TypeError(f"cannot assign into {type(cur).__name__} at {last!r}")
    return new_state


# ---------------------------------------------------------------------------
# Executing a patched service from its SOURCE_CODE string.

def load_service_from_source(source: str, class_name: str):
    """Exec the SOURCE_CODE in a fresh namespace and return the class object.

    The SOURCE_CODE strings in our testbeds are self-contained modules (no
    relative imports). We exec them in a fresh namespace with __builtins__ so
    that standard library imports (math, copy, etc.) still work if needed.
    """
    import builtins, types
    ns = {"__name__": f"__cte_patched_{class_name}__",
          "__builtins__": builtins.__dict__}
    exec(source, ns)
    if class_name not in ns:
        raise KeyError(f"class {class_name} not defined in patched source")
    return ns[class_name]


DOMAIN_CLASS = {
    "bank":         "BankService",
    "cart":         "CartService",
    "auth":         "AuthService",
    "reservation":  "ReservationService",
    "ratelimiter":  "RateLimiter",
    "filesystem":   "FileSystemService",
}
