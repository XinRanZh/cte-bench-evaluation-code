"""Score the identifier-renamed variant study: accuracy and prediction agreement.

Inputs (dataset repository): `counterfactuals/core_v1.jsonl`,
`counterfactuals/core_v1_renamed.jsonl`, and four same-day runs of one model
under answers revealed in `reference_results/renamed_raw/`:
  *_original_run1_*, *_original_run2_*  on the 210 source-edit Core-v1 scenarios
  *_renamed_run1_*,  *_renamed_run2_*   on their renamed copies (paired_item_id)

Reported on the effect steps of the first 40 future calls:
  * effect-step VM of each run (released grader), and the paired difference
    mean(renamed runs) - mean(original runs);
  * exact agreement of predictions between runs: original repeat (o1 vs o2),
    renamed repeat (r1 vs r2), and cross agreement (mean over the four
    original/renamed pairings), with the paired differences
    cross - renamed repeat and renamed repeat - original repeat.
Agreement compares the canonical JSON of the two `predicted` values (a missing
or unparseable prediction is compared as null); oracle traces only select the
effect steps. Calls that returned a provider error without any model output
(`grader.is_infrastructure_failure`) are not scored: they leave the accuracy
of their run and every agreement that involves them; means over runs pool
numerators and denominators. Intervals resample scenarios in item_id order
(B=2000, seed 17, ratio of sums); differences resample both sides jointly.

Each arm is the single file `reference_results/renamed_raw/*_<arm>_*.json`, or
the file given with `--arm ARM=PATH`; several matching files are an error.

Usage:
    python scripts/score_renamed_variant.py --dataset-root path/to/cte-bench-core-v1 --json scratch/renamed.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_provenance as rp  # noqa: E402
import score_paired as sp  # noqa: E402
from grader import grade_value_match, is_infrastructure_failure  # noqa: E402

BUDGET = 40
ARMS = ("original_run1", "original_run2", "renamed_run1", "renamed_run2")
EXCLUDED = object()   # an infrastructure failure: no model output, not scored


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


STUDY = "score_renamed_variant (answers revealed, K=20, 40 calls)"
EXPECT = {"strategy": "sliding", "window": 20, "suffix_budget": 40}


def load_arm(path: Path, catalog: list[dict], catalog_path: Path, catalog_name: str,
             strip: bool) -> tuple[dict, dict]:
    """Predictions of one arm, keyed by the original item_id. Every row is joined
    one-to-one to `catalog` (original arms: the 210 source-edit Core-v1 rows;
    renamed arms: core_v1_renamed.jsonl) and its steps are validated (unique,
    in-range indices; calls equal to the catalog's; targets equal to the
    catalog's responses)."""
    run = json.loads(path.read_text())
    desc = rp.describe_run(run, path.name)
    rp.expect_config(desc, EXPECT, STUDY)
    bound = rp.bind_to_catalog(desc, run, catalog, catalog_path if desc["status"] != rp.LEGACY else None,
                               budget=BUDGET, catalog_name=catalog_name)
    if len(bound) != len(catalog):
        raise sp.InvalidRunError(f"{path.name}: {len(bound)} rows for the {len(catalog)} rows of {catalog_name}")
    out = {}
    excluded = 0
    for _, row in bound:
        iid = row["item_id"].replace("__renamed", "") if strip else row["item_id"]
        preds = [None] * BUDGET
        for step in sp.checked_steps(row, catalog[[c["item_id"] for c in catalog].index(row["item_id"])], BUDGET):
            failed = is_infrastructure_failure(step, row)
            excluded += failed
            preds[step["suffix_idx"]] = EXCLUDED if failed else step.get("predicted")
        out[iid] = preds
    desc["infra_excluded_steps"] = excluded
    return out, desc


def arm_files(root: Path, given: list[str]) -> dict:
    """One file per arm: --arm ARM=PATH, else the single match of *_<arm>_*.json."""
    explicit = {}
    for spec in given:
        arm, sep, path = spec.partition("=")
        if not sep or arm not in ARMS:
            raise SystemExit(f"error: --arm expects ARM=PATH with ARM in {', '.join(ARMS)}; got {spec!r}")
        explicit[arm] = Path(path)
    files = {}
    for arm in ARMS:
        if arm in explicit:
            files[arm] = explicit[arm]
            continue
        hits = sorted((root / "reference_results" / "renamed_raw").glob(f"*_{arm}_*.json"))
        if len(hits) != 1:
            raise SystemExit(f"error: arm {arm}: {len(hits)} files match reference_results/renamed_raw/*_{arm}_*.json "
                             f"({', '.join(h.name for h in hits) or 'none'}); pass --arm {arm}=PATH")
        files[arm] = hits[0]
    return files


def ratio(pairs) -> float:
    den = sum(p[1] for p in pairs)
    return 100.0 * sum(p[0] for p in pairs) / den if den else 0.0


def boot(pairs, n_boot=2000, seed=17):
    rng = random.Random(seed)
    n = len(pairs)
    vals = []
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        vals.append(ratio(s))
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]]


def paired_diff(a, b, n_boot=2000, seed=17):
    """CI for ratio(b) - ratio(a), resampling scenarios once for both sides."""
    rng = random.Random(seed)
    n = len(a)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        vals.append(ratio([b[i] for i in idx]) - ratio([a[i] for i in idx]))
    vals.sort()
    return [vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    ap.add_argument("--arm", action="append", default=[], metavar="ARM=PATH",
                    help="file of one arm (default: the single reference_results/renamed_raw/*_<arm>_*.json)")
    args = ap.parse_args()
    root = args.dataset_root
    core = {r["item_id"]: r for r in sp.load_jsonl(root / "counterfactuals" / "core_v1.jsonl")}
    renamed = sp.load_jsonl(root / "counterfactuals" / "core_v1_renamed.jsonl")
    ids = [r["paired_item_id"] for r in renamed]
    eff, target = {}, {}
    for iid in ids:
        c = core[iid]
        o = [s["response"] for s in c["trace_original"]][:BUDGET]
        t = [s["response"] for s in c["trace_counterfactual"]][:BUDGET]
        eff[iid] = [i for i in range(len(t)) if canon(o[i]) != canon(t[i])]
        target[iid] = t
    files = arm_files(root, args.arm)
    originals = [c for c in sp.load_jsonl(root / "counterfactuals" / "core_v1.jsonl") if c["item_id"] in set(ids)]
    catalogs = {False: (originals, root / "counterfactuals" / "core_v1.jsonl", "the 210 source-edit Core-v1 rows"),
                True: (renamed, root / "counterfactuals" / "core_v1_renamed.jsonl", "core_v1_renamed.jsonl")}
    try:
        loaded = {arm: load_arm(path, *catalogs[arm.startswith("renamed")], strip=arm.startswith("renamed"))
                  for arm, path in files.items()}
    except sp.InvalidRunError as e:
        raise SystemExit(f"error: invalid arm: {e}")
    preds = {arm: v[0] for arm, v in loaded.items()}
    descs = [v[1] for v in loaded.values()]
    # The four arms differ only in the catalog (original vs renamed source).
    try:
        prov = rp.check(descs, allow_mixed=args.allow_mixed, tool="score_renamed_variant",
                        extra_problems=rp.combination_problems(descs))
    except rp.MixedRunError as e:
        raise SystemExit(f"error: {e}")
    for arm, p in preds.items():
        if set(p) != set(ids):
            raise SystemExit(f"{files[arm].name}: covers {len(p)} scenarios, expected the {len(ids)} renamed ones")
    scen = sorted(i for i in ids if eff[i])  # scenarios with effect steps, sorted by item_id

    def acc_pairs(arm):
        out = []
        for i in scen:
            js = [j for j in eff[i] if preds[arm][i][j] is not EXCLUDED]
            out.append((sum(bool(grade_value_match(preds[arm][i][j], target[i][j])["match"]) for j in js), len(js)))
        return out

    def agree_pairs(a, b):
        out = []
        for i in scen:
            js = [j for j in eff[i] if preds[a][i][j] is not EXCLUDED and preds[b][i][j] is not EXCLUDED]
            out.append((sum(canon(preds[a][i][j]) == canon(preds[b][i][j]) for j in js), len(js)))
        return out

    lead = rp.report_fields(loaded["original_run1"][1])
    protocol = (f"tf1step_sliding_k{lead['window']}" if lead["strategy"] == "sliding" else str(lead["strategy"]))
    out = {"model": json.loads(files["original_run1"].read_text()).get("model"),
           "protocol": protocol if not prov["mixed"] else "MIXED (see arms)", "scenarios": len(ids), "scenarios_with_effect_steps": len(scen),
           "effect_steps": sum(len(eff[i]) for i in scen),
           "bootstrap": "scenario resampling, B=2000, seed 17, ratio of sums",
           "files": {arm: f"reference_results/renamed_raw/{f.name}" for arm, f in files.items()},
           "effect_step_vm": {}, "agreement": {},
           "infrastructure_failures_excluded": {**{arm: v[1]["infra_excluded_steps"] for arm, v in loaded.items()},
                                                "rule": sp.INFRA_RULE}}
    if rp.needs_report(prov):
        out["provenance"] = prov
        out["arms"] = {arm: rp.report_fields(v[1]) for arm, v in loaded.items()}
        out["model"] = out["arms"]["original_run1"]["model"]
        if prov["mixed"]:
            out["warning"] = rp.warning_line(prov)
            print(out["warning"])
    acc = {arm: acc_pairs(arm) for arm in ARMS}
    for arm in ARMS:
        out["effect_step_vm"][arm] = {"value": ratio(acc[arm]), "ci95": boot(acc[arm])}
    # means over runs pool numerators and denominators (equal denominators unless a call was excluded)
    mean_o = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(acc["original_run1"], acc["original_run2"])]
    mean_r = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(acc["renamed_run1"], acc["renamed_run2"])]
    out["effect_step_vm"]["renamed_minus_original"] = {"value": ratio(mean_r) - ratio(mean_o),
                                                       "ci95": paired_diff(mean_o, mean_r)}
    oo = agree_pairs("original_run1", "original_run2")
    rr = agree_pairs("renamed_run1", "renamed_run2")
    cross_list = [agree_pairs(a, b) for a, b in itertools.product(ARMS[:2], ARMS[2:])]
    cross = [(sum(cl[k][0] for cl in cross_list) / 4, sum(cl[k][1] for cl in cross_list) / 4)
             for k in range(len(scen))]
    for name, pairs in (("original_repeat", oo), ("renamed_repeat", rr), ("cross", cross)):
        out["agreement"][name] = {"value": ratio(pairs), "ci95": boot(pairs)}
    out["agreement"]["cross_minus_renamed_repeat"] = {"value": ratio(cross) - ratio(rr), "ci95": paired_diff(rr, cross)}
    out["agreement"]["renamed_minus_original_repeat"] = {"value": ratio(rr) - ratio(oo), "ci95": paired_diff(oo, rr)}
    for k, v in out["effect_step_vm"].items():
        print(f"effect-step VM {k:24s} {v['value']:6.2f}  [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
    for k, v in out["agreement"].items():
        print(f"agreement      {k:28s} {v['value']:6.2f}  [{v['ci95'][0]:.2f}, {v['ci95'][1]:.2f}]")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
