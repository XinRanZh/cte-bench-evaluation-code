"""Run provenance: the one validation path for every script that reads evaluator outputs.

`eval_llm_counterfactual.py` writes, for every output row, `run_config` (model,
strategy, window, prompt version, budgets, reasoning, phases, catalog sha256)
and `catalog_row_sha256` (the identity of the catalog row it answers), plus a
top-level `run_config` and `mixed_run_configurations`. Outputs written before
stamping existed carry none of these ("unstamped legacy"); the released
reference outputs are legacy.

Classification of one output (`describe_run`):
  legacy    no provenance at all (no top-level run_config / prompt_version /
            mixed flag, no row stamps);
  stamped   every row carries the same run_config, and every declaration agrees
            with it: the top-level run_config and the top-level model, strategy,
            window, prompt_version, thinking, reasoning_effort, max_tokens and
            suffix_budget;
  mixed     marked mixed by the evaluator, rows with different stamps, or a
            declaration that contradicts the stamps (or a study expectation);
  damaged   modern provenance with missing or partial row stamps and no mixed
            marker. Always rejected, also with --allow-mixed.

Consumers bind every input to its catalog (`bind_to_catalog`: catalog hash,
one-to-one row join, per-step validation with score_paired.checked_steps),
apply study expectations (`expect_config`), check combinations of inputs
(`combination_problems`: legacy is never combined with stamped; stamped inputs
must agree on the compared fields), and call `check`: any problem is an error
unless --allow-mixed, in which case the report keeps every configuration and
carries `mixed_run_provenance` fields and a WARNING line. Reported settings come
from the stamps, or from the validated declarations of legacy files. Clean
legacy inputs produce exactly the reports they produced before.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROW_CONFIG_KEY = "run_config"            # = eval_llm_counterfactual.ROW_CONFIG_KEY
ROW_ID_KEY = "catalog_row_sha256"        # = eval_llm_counterfactual.ROW_ID_KEY
MIXED_FLAG = "mixed_run_configurations"  # top-level flag written by the evaluator
CONFIG_FIELDS = ("model", "strategy", "window", "prompt_version", "suffix_budget",
                 "thinking", "reasoning_effort", "max_tokens", "explain", "state",
                 "request_settings", "cf_jsonl_sha256")
COMPARE_FIELDS = CONFIG_FIELDS[:-1]       # combinations: the catalog may differ by design
DECLARED_FIELDS = ("model", "strategy", "window", "prompt_version", "thinking",
                   "reasoning_effort", "max_tokens", "suffix_budget")
REPORT_FIELDS = ("model", "strategy", "window", "prompt_version", "thinking",
                 "reasoning_effort", "max_tokens", "suffix_budget", "request_settings", "cf_jsonl_sha256")
MODERN_TOP_KEYS = ("run_config", MIXED_FLAG, "prompt_version")
STAMPED, LEGACY, MIXED, DAMAGED = "stamped", "unstamped legacy", "mixed", "damaged"
ALLOW_FLAG_HELP = ("Proceed with inputs whose run provenance is mixed (rows or files from "
                   "different run configurations, declarations that contradict the stamps, "
                   "settings other than the study's, stamped with unstamped inputs, or another "
                   "catalog). The report then lists every configuration and is labelled MIXED. "
                   "Damaged inputs (partial stamps) are always rejected.")


# Model identities. Each entry lists the `model` fields that name one model in
# the released reference outputs (API identifiers and the labels of the
# earlier runner). Two outputs are the same model only if their fields map to
# the same identity; a field not listed here is its own identity.
MODEL_ALIASES = {
    "DeepSeek V4-Flash": ("deepseek/deepseek-chat", "DS-V4-Lite/sliding"),
    "DeepSeek V4-Flash (reasoning on)": ("deepseek/deepseek-reasoner",),
    "Kimi K2.5": ("bedrock/moonshotai.kimi-k2.5", "Kimi-K2.5/sliding", "Kimi-K2.5/prefix_only",
                  "kimi-k2.5/sliding"),
    "Qwen3.6-35B-A3B": ("openrouter/qwen/qwen3.6-35b-a3b",),
    "Claude Sonnet 4.6": ("bedrock/global.anthropic.claude-sonnet-4-6", "bedrock/us.anthropic.claude-sonnet-4-6",
                          "anthropic/claude-sonnet-4-6", "sonnet-4.6/sliding"),
    "Claude Opus 4.7": ("opus-4.7/sliding",),
}
_IDENTITY = {alias: name for name, aliases in MODEL_ALIASES.items() for alias in aliases}


def model_identity(model) -> str:
    """The model a `model` field names (MODEL_ALIASES), else the field itself."""
    return _IDENTITY.get(model, str(model))


class MixedRunError(ValueError):
    """Inputs with mixed or damaged run provenance."""


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def catalog_row_id(row: dict) -> str:
    """Unambiguous identity of one catalog row (its full content)."""
    return hashlib.sha256(canon(row).encode()).hexdigest()


# ---------------------------------------------------------------------------
# One output

def describe_run(run: dict, name: str) -> dict:
    rows = [r for r in run.get("per_item", []) if isinstance(r, dict)]
    stamps = [r.get(ROW_CONFIG_KEY) for r in rows]
    stamped = [s for s in stamps if isinstance(s, dict)]
    n_unstamped = len(rows) - len(stamped)
    modern = bool(stamped) or any(k in run for k in MODERN_TOP_KEYS)
    configs: dict = {}
    for s in stamps:
        key = canon(s if isinstance(s, dict) else None)
        configs[key] = configs.get(key, 0) + 1
    desc = {"file": name, "n_rows": len(rows), "n_unstamped_rows": n_unstamped,
            "row_configurations": [{"run_config": json.loads(k), "n_rows": n} for k, n in configs.items()],
            "declared": {k: run.get(k) for k in DECLARED_FIELDS},
            "config": None, "problems": [], "damage": []}
    if not modern:
        desc["status"] = LEGACY
        return desc
    flagged = bool(run.get(MIXED_FLAG))
    if n_unstamped and not flagged:
        desc["damage"].append(f"{n_unstamped} of its {len(rows)} rows carry no run_config although the file "
                              "carries modern provenance and no mixed marker")
    if flagged:
        desc["problems"].append("the evaluator marked this output mixed_run_configurations=true")
    if stamped and n_unstamped:
        desc["problems"].append(f"{n_unstamped} unstamped rows are combined with {len(stamped)} stamped rows")
    distinct = {canon(s) for s in stamped}
    if len(distinct) > 1:
        desc["problems"].append(f"its rows carry {len(distinct)} different run configurations")
    if stamped and len(distinct) == 1:
        cfg = stamped[0]
        desc["config"] = cfg
        top = run.get("run_config")
        if isinstance(top, dict) and canon(top) != canon(cfg) and not flagged:
            diffs = [k for k in CONFIG_FIELDS if top.get(k) != cfg.get(k)]
            desc["problems"].append(f"its top-level run_config differs from its row stamps in {diffs}")
        for k in DECLARED_FIELDS:
            if k in run and run.get(k) != cfg.get(k):
                desc["problems"].append(f"it declares {k}={run.get(k)!r} but its rows were produced with "
                                        f"{k}={cfg.get(k)!r}")
    elif not stamped:
        desc["damage"].append("it carries modern provenance but no row stamps")
    desc["status"] = DAMAGED if desc["damage"] else (MIXED if desc["problems"] else STAMPED)
    return desc


def report_fields(desc: dict) -> dict:
    """Settings to report for one input: from the stamps, else the legacy declarations."""
    if desc["status"] == STAMPED:
        return {k: desc["config"].get(k) for k in REPORT_FIELDS}
    if desc["status"] == LEGACY:
        return {k: desc["declared"].get(k) for k in REPORT_FIELDS if k != "cf_jsonl_sha256"} | {"cf_jsonl_sha256": None}
    out = {}
    for k in REPORT_FIELDS:
        vals = {canon(c["run_config"].get(k) if c["run_config"] else desc["declared"].get(k))
                for c in desc["row_configurations"]}
        out[k] = json.loads(vals.pop()) if len(vals) == 1 else "MIXED"
    return out


def config_key(desc: dict) -> str:
    """Complete configuration key used for grouping (legacy files never share a key)."""
    if desc["status"] == LEGACY:
        return canon({"legacy_file": desc["file"], **desc["declared"]})
    return canon(report_fields(desc))


def _mark(desc: dict, problem: str) -> None:
    desc["problems"].append(problem)
    if desc["status"] == STAMPED:
        desc["status"] = MIXED


def expect_config(desc: dict, expected: dict, study: str) -> None:
    """Study-specific settings. expected: field -> allowed value or tuple of values.
    Legacy files are checked on the fields they declare."""
    have = report_fields(desc) if desc["status"] != LEGACY else desc["declared"]
    for k, allowed in expected.items():
        allowed = allowed if isinstance(allowed, tuple) else (allowed,)
        v = have.get(k)
        if desc["status"] == LEGACY and v is None:
            continue
        if v not in allowed:
            _mark(desc, f"{study} expects {k} in {list(allowed)} but the input has {k}={v!r}")


def bind_to_catalog(desc: dict, run: dict, catalog: list[dict], catalog_path=None, *,
                    budget: int = 40, catalog_name: str = "the catalog", require_complete: bool = True,
                    join: str = "auto") -> list[tuple[int, dict]]:
    """Join every row one-to-one to a catalog row and validate its steps.

    Stamped rows join on catalog_row_sha256 and their run_config's catalog hash
    must equal the catalog's. Legacy rows join on item_id when that id occurs
    once in the catalog, else on the full content of the scored calls and
    targets among the catalog rows with that id. A row that answers no catalog
    row, or two rows that answer the same one, raise InvalidRunError.
    Returns [(catalog_index, row)] in catalog order."""
    import score_paired as sp  # local import: score_paired imports this module
    if desc["status"] == STAMPED and catalog_path is not None:
        got = desc["config"].get("cf_jsonl_sha256")
        if got != file_sha256(catalog_path):
            _mark(desc, f"its rows were produced on another catalog than {catalog_name}")
    rows = [r for r in run.get("per_item", []) if isinstance(r, dict) and "error" not in r]
    ids = [r.get("item_id") for r in catalog]
    by_uid = {catalog_row_id(c): i for i, c in enumerate(catalog)}
    by_id: dict = {}
    for i, iid in enumerate(ids):
        by_id.setdefault(iid, []).append(i)
    content = {}
    for i, c in enumerate(catalog):
        key = tuple((s["op"], canon(s["args"]), canon(s["response"])) for s in c["trace_counterfactual"][:budget])
        content.setdefault(key, []).append(i)
    out, seen = [], {}
    for pos, r in enumerate(rows):
        if r.get(ROW_ID_KEY):
            i = by_uid.get(r[ROW_ID_KEY])
            if i is None:
                raise sp.InvalidRunError(f"{desc['file']}: row {pos} ({r.get('item_id')}) answers no row of {catalog_name}")
        elif r.get("item_id") not in by_id:
            raise sp.InvalidRunError(f"{desc['file']}: row {pos} item_id {r.get('item_id')!r} is not in {catalog_name}")
        elif len(by_id[r["item_id"]]) == 1 and join != "content":
            i = by_id[r["item_id"]][0]
        else:
            key = tuple((s.get("op"), canon(s.get("args")), canon(s.get("ground_truth")))
                        for s in sp.scored_steps(r, budget))
            hits = [h for h in content.get(key, []) if h in by_id[r["item_id"]]]
            if len(hits) != 1:
                raise sp.InvalidRunError(f"{desc['file']}: row {pos} ({r.get('item_id')}) matches {len(hits)} rows "
                                         f"of {catalog_name} by content")
            i = hits[0]
        if i in seen:
            raise sp.InvalidRunError(f"{desc['file']}: rows {seen[i]} and {pos} both answer catalog row {i} "
                                     f"({ids[i]}) of {catalog_name}")
        seen[i] = pos
        c = catalog[i]
        if r.get("item_id") != c["item_id"]:
            raise sp.InvalidRunError(f"{desc['file']}: row {pos} carries item_id {r.get('item_id')!r} but answers "
                                     f"{c['item_id']!r}")
        steps = sp.checked_steps(r, c, budget)
        for s in steps:
            want = c["trace_counterfactual"][s["suffix_idx"]]["response"]
            if "ground_truth" in s and canon(s["ground_truth"]) != canon(want):
                raise sp.InvalidRunError(f"{desc['file']}: row {pos} ({c['item_id']}) step {s['suffix_idx']} stores a "
                                         f"target that is not the catalog response")
        if require_complete and len(steps) != min(budget, len(c["trace_counterfactual"])):
            raise sp.InvalidRunError(f"{desc['file']}: row {pos} ({c['item_id']}) has {len(steps)} scored steps")
        out.append((i, r))
    return sorted(out, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Combinations and the decision

def combination_problems(descs: list[dict], fields=COMPARE_FIELDS) -> list[str]:
    problems = []
    stamped = [d for d in descs if d["status"] in (STAMPED, MIXED) and d["config"] is not None]
    legacy = [d for d in descs if d["status"] == LEGACY]
    if legacy and len(legacy) < len(descs):
        problems.append("unstamped legacy input(s) " + ", ".join(d["file"] for d in legacy) +
                        " are combined with stamped input(s) " +
                        ", ".join(d["file"] for d in descs if d["status"] != LEGACY) +
                        "; their configurations cannot be verified to agree")
    for d in stamped[1:]:
        ref = stamped[0]
        diffs = [f"{k}={d['config'].get(k)!r} vs {ref['config'].get(k)!r}" for k in fields
                 if d["config"].get(k) != ref["config"].get(k)]
        if diffs:
            problems.append(f"{d['file']} and {ref['file']} were produced with different run "
                            f"configurations ({'; '.join(diffs)})")
    return problems


def catalog_problems(desc: dict, catalog_sha: str | None, catalog_name: str) -> list[str]:
    if desc["status"] not in (STAMPED, MIXED) or desc["config"] is None or catalog_sha is None:
        return []
    if desc["config"].get("cf_jsonl_sha256") != catalog_sha:
        return [f"{desc['file']} was produced on another catalog than {catalog_name}"]
    return []


def check(descs: list[dict], *, allow_mixed: bool, tool: str, extra_problems=()) -> dict:
    """Raise unless the inputs are clean or allow_mixed; damaged inputs always raise."""
    damage = [f"{d['file']}: {p}" for d in descs for p in d["damage"]]
    if damage:
        raise MixedRunError(f"{tool}: damaged run provenance: " + " | ".join(damage) +
                            ". Only outputs without any provenance are treated as unstamped legacy.")
    problems = [f"{d['file']}: {p}" for d in descs for p in d["problems"]] + list(extra_problems)
    if problems and not allow_mixed:
        raise MixedRunError(f"{tool}: refusing inputs with mixed run provenance: " + " | ".join(problems) +
                            ". Pass --allow-mixed to proceed; the report is then labelled MIXED.")
    return {
        "mixed": bool(problems),
        "problems": problems,
        "inputs": [{"file": d["file"], "status": d["status"], "n_rows": d["n_rows"],
                    "n_unstamped_rows": d["n_unstamped_rows"], "row_configurations": d["row_configurations"]}
                   for d in descs],
    }


def needs_report(prov: dict) -> bool:
    """Whether a report must carry provenance fields (anything but clean legacy input)."""
    return prov["mixed"] or any(i["status"] != LEGACY for i in prov["inputs"])


def warning_line(prov: dict) -> str:
    return ("WARNING: MIXED RUN PROVENANCE (--allow-mixed): " + " | ".join(prov["problems"])
            if prov["mixed"] else "")
