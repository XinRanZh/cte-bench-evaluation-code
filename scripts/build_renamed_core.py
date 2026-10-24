"""Build a trace-verified identifier-randomized Core-v1 variant.

Purpose: three of the six services port algorithms from well-known libraries
(django-oscar, python-limits, pyfilesystem2), so a score could in part reflect
familiarity with their API surface. This script rewrites the identifier surface
while leaving program behaviour unchanged on the evaluated traces.

What a score difference means: renaming removes library-specific API names and
also generic semantic cues that a simulator may legitimately use (`withdraw`
implies a balance decrease). A difference therefore measures sensitivity to
identifier semantics in general, not recall of a particular library.

What is renamed (Python tokenizer, so never inside strings or comments):
  - the service class name
  - every public operation name (these are also the trace `op` names)
  - private helper method names
  - instance attribute names

What is deliberately NOT renamed:
  - response dict keys and error-code strings (they are response VALUES, so
    keeping them makes the oracle byte-identical to the published catalog and
    makes verification exact)
  - call arguments, which the harness passes positionally on TypeError
  - line structure: `tokenize.untokenize` preserves the line count, so every
    published `intervention.line_number` still points at the patched line

Verification (`--verify`, enabled by default): for every item the renamed
program is executed under the paper's own do-calculus oracle semantics -- replay
the prefix on the ORIGINAL class, swap `__class__` to the patched class, replay
the suffix -- and each response must equal the response produced by the
identical unrenamed replay. Comparing against the identity replay rather than
against the catalog isolates the rename's effect from any pre-existing catalog
property.

This is trace verification, not a proof: it establishes output equality on the
evaluated traces, not over all reachable states and inputs. Report it as
"zero output mismatches across N paired replay steps", never as "provably
semantics-preserving".

Usage:
    python3 scripts/build_renamed_core.py --out counterfactuals/core_v1_renamed.jsonl
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from testbeds import TESTBEDS  # noqa: E402
from interventions import load_service_from_source  # noqa: E402
from gen_counterfactuals import DOMAIN_CLASS, _replay  # noqa: E402

CATALOG = ROOT / "counterfactuals" / "core_v1.jsonl"
SUFFIX_BUDGET = 40


def build_map(domain: str) -> tuple[str, str, dict[str, str]]:
    """Bijective identifier map for one service. Neutral, semantics-free names."""
    source = TESTBEDS[domain].SOURCE_CODE
    cls = DOMAIN_CLASS[domain]
    methods = sorted(set(re.findall(r"\n    def (\w+)", source)) - {"__init__"})
    attrs = sorted(set(re.findall(r"self\.(\w+)", source)) - set(methods))
    mapping = {name: f"op{i:02d}" for i, name in enumerate(methods)}
    mapping.update({name: f"st{i:02d}" for i, name in enumerate(attrs)})
    mapping.pop("self", None)
    mapping[cls] = f"Svc{domain[:3].upper()}"
    return source, cls, mapping


SKIP_TOKENS = (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
               tokenize.DEDENT, tokenize.COMMENT)


def rename_source(source: str, mapping: dict[str, str]) -> str:
    """Context-aware token rename.

    A NAME is rewritten only when it is a bare identifier or a `self.<name>`
    attribute. Attribute access on any other object is left alone -- otherwise a
    builtin call like `self.inodes[x].remove(y)` would be renamed just because
    the service happens to define its own `remove` operation.
    """
    if not mapping:
        return source
    out: list[tuple[int, str]] = []
    significant: list[tokenize.TokenInfo] = []
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        text = tok.string
        if tok.type == tokenize.NAME and text in mapping:
            after_dot = bool(significant) and significant[-1].string == "."
            on_self = after_dot and len(significant) >= 2 and significant[-2].string == "self"
            if not after_dot or on_self:
                text = mapping[text]
        out.append((tok.type, text))
        if tok.type not in SKIP_TOKENS:
            significant.append(tok)
    return tokenize.untokenize(out)


def rename_value(value, mapping: dict[str, str]):
    """Rewrite identifier occurrences inside an intervention span string."""
    if not isinstance(value, str):
        return value
    for old in sorted(mapping, key=len, reverse=True):
        value = re.sub(rf"\b{re.escape(old)}\b", mapping[old], value)
    return value


def make_factory(source: str, cls_name: str, ctor_kwargs: dict):
    cls = load_service_from_source(source, cls_name)

    def make():
        try:
            return cls()
        except TypeError:
            return cls(**ctor_kwargs)

    return make


def run_oracle(item: dict, mapping: dict[str, str], factory, cls_name: str):
    """Replay the paper's do-calculus oracle under `mapping`."""
    service = factory()
    prefix = _replay(service, [{"op": mapping.get(s["op"], s["op"]), "args": s["args"]}
                               for s in item["trace_prefix"]])
    patched = load_service_from_source(
        rename_source(item["patched_source"], mapping), cls_name)
    service.__class__ = patched
    suffix = _replay(service, [{"op": mapping.get(s["op"], s["op"]), "args": s["args"]}
                               for s in item["trace_counterfactual"][:SUFFIX_BUDGET]])
    return ([s["response"] for s in prefix], [s["response"] for s in suffix])


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "counterfactuals" / "core_v1_renamed.jsonl")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    rows = [json.loads(line) for line in CATALOG.open()]
    by_domain = collections.defaultdict(list)
    for row in rows:
        by_domain[row["domain"]].append(row)

    maps, factories = {}, {}
    for domain in sorted(by_domain):
        source, cls, mapping = build_map(domain)
        reference = TESTBEDS[domain].new_service()
        ctor = {k: getattr(reference, k) for k in ("limit", "expiry")
                if hasattr(reference, k)}
        maps[domain] = (source, cls, mapping)
        factories[domain] = (
            make_factory(rename_source(source, mapping), mapping[cls],
                         {mapping.get(k, k): v for k, v in ctor.items()}),
            make_factory(source, cls, ctor),
        )

    verified_steps = 0
    mismatches = 0
    deferred = 0
    out_rows = []

    for domain in sorted(by_domain):
        source, cls, mapping = maps[domain]
        renamed_factory, identity_factory = factories[domain]
        new_cls = mapping[cls]
        dom_steps = dom_bad = dom_deferred = 0

        for item in by_domain[domain]:
            if not item.get("patched_source"):
                # state_injection / clock_jump items carry no source patch; the
                # oracle path differs, so the variant covers source edits only.
                dom_deferred += 1
                continue

            if not args.no_verify:
                got_prefix, got_suffix = run_oracle(item, mapping, renamed_factory, new_cls)
                exp_prefix, exp_suffix = run_oracle(item, {}, identity_factory, cls)
                for got, exp in list(zip(got_prefix, exp_prefix)) + list(zip(got_suffix, exp_suffix)):
                    dom_steps += 1
                    if canon(got) != canon(exp):
                        dom_bad += 1

            new_item = dict(item)
            new_item["item_id"] = item["item_id"] + "__renamed"
            new_item["source_variant"] = "identifier_randomized"
            new_item["paired_item_id"] = item["item_id"]
            new_item["patched_source"] = rename_source(item["patched_source"], mapping)
            new_item["identifier_map"] = mapping
            new_item["intervention"] = {
                k: (rename_value(v, mapping) if k in ("original_span", "modified_span", "description") else v)
                for k, v in item["intervention"].items()
            }
            for trace_key in ("trace_prefix", "trace_original", "trace_counterfactual"):
                new_item[trace_key] = [
                    {**step, "op": mapping.get(step["op"], step["op"])}
                    for step in item[trace_key]
                ]
            out_rows.append(new_item)

        verified_steps += dom_steps
        mismatches += dom_bad
        deferred += dom_deferred
        status = "SKIP" if args.no_verify else ("OK  " if dom_bad == 0 else "FAIL")
        print(f"{domain:12s} {status} renamed={len(mapping):3d} identifiers | "
              f"items={len(by_domain[domain]) - dom_deferred:3d} "
              f"(state-intervention deferred={dom_deferred:2d}) "
              f"steps verified={dom_steps:5d} mismatches={dom_bad}")

    # --no-verify skips the trace check: the catalog is then unverified, never "verified"
    trace_verified = (not args.no_verify) and mismatches == 0
    print(f"\nTOTAL verified steps={verified_steps}  mismatches={mismatches}")
    if args.no_verify:
        print("TRACE-VERIFIED: skipped (--no-verify); the written catalog is UNVERIFIED")
    else:
        print(f"TRACE-VERIFIED (output equality on evaluated traces): {trace_verified}")
    print(f"renamed items written={len(out_rows)}  state-intervention items excluded={deferred}")

    if not args.no_verify and not trace_verified:
        print("\nrefusing to write a catalog with any output mismatch")
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for row in out_rows:
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {args.out} ({len(out_rows)} items)")

    if args.report:
        args.report.write_text(json.dumps({
            **({"verification": "skipped (--no-verify): the catalog is unverified"} if args.no_verify else {}),
            "verified_steps": verified_steps,
            "mismatches": mismatches,
            "trace_verified": trace_verified,
            "items_written": len(out_rows),
            "state_intervention_items_excluded": deferred,
            "identifiers_renamed": {d: len(m) for d, (_, _, m) in maps.items()},
        }, indent=2))
        print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
