"""Post-hoc application of reachability filter v2 and state-probe v2.

No model calls. Takes (a) the catalog the results were produced on and (b)
existing result JSONs, and produces:

  - A per-row filter verdict file with n_diff, first_diff_step, and
    (for state items) reachability-v2 reason.
  - A re-aggregated leaderboard where rows that v2 rejects are removed.

The catalog is given explicitly with --catalog (one or more JSONL files, e.g.
counterfactuals/core_v1.jsonl: every Core-v1 row passes the filter), or with
--cf-dir for a development candidate pool (every *.jsonl in the directory
except core_v1.jsonl, which is a filtered subset of the pool). Every result
row must answer exactly one catalog row (id, calls, targets); a result file
with rows outside the catalog stops the run instead of reporting them as
rejected.

Usage:
  python scripts/apply_v2_filter.py \\
      --catalog counterfactuals/core_v1.jsonl \\
      --results-dir path/to/cte-bench-core-v1/reference_results/raw \\
      --out-dir scratch/v2_filter
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reachability import check_suffix_causal_effect
import run_provenance as rp  # noqa: E402
import score_paired as sp  # noqa: E402
import grader  # noqa: E402


def _row_key(item: dict, row_index: int) -> tuple:
    """Globally unique key for one counterfactual row.

    item_id alone is NOT unique, and semantic keys can still collide for
    repeated adversarial-patch clusters. The catalog row index is the stable
    unit of evaluation because result JSONs preserve catalog order.
    """
    iv = item.get("intervention") or {}
    return (
        row_index,
        item.get("item_id"),
        item.get("domain"),
        iv.get("kind"),
        item.get("prefix_len"),
        str(iv.get("from_value")),
        str(iv.get("to_value")),
        str(iv.get("line_number") if "line_number" in iv else iv.get("path")),
    )


def catalog_files(cf_dir: Path | None, catalogs: list[Path]) -> list[Path]:
    if catalogs:
        return list(catalogs)
    return [jf for jf in sorted(cf_dir.glob("*.jsonl")) if jf.name != "core_v1.jsonl"]


def load_catalog(files: list[Path]) -> dict[tuple, dict]:
    """Load the catalog rows (in order) keyed by a globally unique row tuple."""
    out = {}
    row_index = 0
    for jf in files:
        for line in jf.open():
            try:
                item = json.loads(line)
            except Exception:
                continue
            k = _row_key(item, row_index)
            out[k] = item
            row_index += 1
    return out


def classify_items(catalog: dict[tuple, dict]) -> dict[tuple, dict]:
    """For every row, return {kept: bool, reason, n_diff, first_diff_step}."""
    out = {}
    for k, item in catalog.items():
        verdict = check_suffix_causal_effect(item)
        out[k] = {
            "kept": bool(verdict.accepted),
            "reason": verdict.reason,
            **verdict.detail,
        }
    return out


CATALOG_SHAS: set[str] = set()  # sha256 of the catalog files (set in main)
CATALOG_ROWS: dict = {}           # row key -> catalog row (set in main)


def is_result_file(path: Path) -> bool:
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return isinstance(d, dict) and isinstance(d.get("per_item"), list)


def aggregate_filtered(result_path: Path, v2_filter: dict[tuple, dict], allow_mixed: bool = False) -> dict:
    """Recompute per-model totals, excluding rows v2 rejects.

    Every result row is bound one-to-one to its catalog row (id, calls,
    targets) with run_provenance.bind_to_catalog; a row outside the catalog
    raises instead of being counted as rejected."""
    d = json.loads(result_path.read_text())
    desc = rp.describe_run(d, result_path.name)
    if desc["config"] is not None and CATALOG_SHAS and desc["config"].get("cf_jsonl_sha256") not in CATALOG_SHAS:
        rp._mark(desc, "its rows were produced on none of the --cf-dir catalog files")
    prov = rp.check([desc], allow_mixed=allow_mixed, tool="apply_v2_filter")
    per_item = [r for r in d.get("per_item", []) if r and "error" not in r]

    keys = sorted(v2_filter.keys(), key=lambda k: k[0])
    catalog_rows = [CATALOG_ROWS[k] for k in keys]
    bound = rp.bind_to_catalog(desc, d, catalog_rows, None, catalog_name="the given catalog",
                               require_complete=False)
    kept = [r for i, r in bound if v2_filter[keys[i]].get("kept")]
    dropped = [r for i, r in bound if not v2_filter[keys[i]].get("kept")]

    def summarize(items):
        if not items:
            return None
        # calls that returned a provider error without model output are not scored
        items = [grader.scoring_view(r) for r in items]
        infra = sum(sum((r.get("n_infra_excluded") or {}).values()) for r in items)
        suffix_total = sum(r["suffix_len"] for r in items)
        vm = sum(r["n_value_match"] for r in items)
        sm = sum(r["n_status_match"] for r in items)
        wf = sum(r["n_well_formed"] for r in items)
        adv = [r for r in items if r.get("intervention_kind") == "adversarial"]
        adv_suffix = sum(r["suffix_len"] for r in adv)
        adv_vm = sum(r["n_value_match"] for r in adv)
        # expl (phase_b)
        expl_items = [r for r in items
                      if r.get("phase_b") and r["phase_b"].get("applicable")]
        expl_exact = sum(1 for r in expl_items if r["phase_b"].get("exact_match"))
        expl_adj = sum(1 for r in expl_items if r["phase_b"].get("adjacent_match"))
        # state probe v1
        sp_v1_items = [r for r in items
                       if r.get("phase_c") and r["phase_c"].get("valid")]
        sp_v1_paths = sum(r["phase_c"]["n_paths"] for r in sp_v1_items)
        sp_v1_match = sum(r["phase_c"]["n_match"] for r in sp_v1_items)
        # state probe v2 (if populated)
        sp_v2_items = [r for r in items
                       if r.get("phase_c") and (r["phase_c"].get("v2") or {}).get("valid")]
        sp_v2_eff = sum((r["phase_c"]["v2"] or {}).get("n_effective", 0)
                        for r in sp_v2_items)
        sp_v2_match = sum((r["phase_c"]["v2"] or {}).get("n_match", 0)
                          for r in sp_v2_items)
        return {
            "n_items": len(items),
            "suffix_total": suffix_total,
            "vm": (vm / suffix_total) if suffix_total else None,
            "sm": (sm / suffix_total) if suffix_total else None,
            "wf": (wf / suffix_total) if suffix_total else None,
            "adv_vm": (adv_vm / adv_suffix) if adv_suffix else None,
            "n_expl_applicable": len(expl_items),
            "expl_exact": (expl_exact / len(expl_items)) if expl_items else None,
            "expl_adj": (expl_adj / len(expl_items)) if expl_items else None,
            "n_sp_v1_valid": len(sp_v1_items),
            "sp_v1_fraction": (sp_v1_match / sp_v1_paths) if sp_v1_paths else None,
            "n_sp_v2_valid": len(sp_v2_items),
            "sp_v2_fraction": (sp_v2_match / sp_v2_eff) if sp_v2_eff else None,
            "sp_v2_n_effective": sp_v2_eff,
            "infra_excluded_calls": infra,
        }

    provenance = {}
    if rp.needs_report(prov):
        provenance = {**rp.report_fields(desc), "provenance": prov}
        if prov["mixed"]:
            provenance["warning"] = rp.warning_line(prov)
            print(provenance["warning"])
    return {
        "model": d.get("model"),
        **provenance,
        "n_items_before_v2": len(per_item),
        "n_items_after_v2": len(kept),
        "n_items_dropped_by_v2": len(dropped),
        "summary_all_items": summarize(per_item),
        "summary_after_v2_filter": summarize(kept),
        "summary_of_dropped_items": summarize(dropped),
    }


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--catalog", action="append", type=Path, default=[],
                     help="catalog JSONL the results were produced on (repeatable), e.g. "
                          "counterfactuals/core_v1.jsonl")
    src.add_argument("--cf-dir", type=Path,
                     help="development candidate pool: every *.jsonl in the directory except "
                          "core_v1.jsonl")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="directory of per-model result JSONs")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = catalog_files(args.cf_dir, args.catalog)
    CATALOG_SHAS.update(rp.file_sha256(f) for f in files)
    print(f"loading item catalog from {', '.join(str(f) for f in files)} ...")
    catalog = load_catalog(files)
    CATALOG_ROWS.update(catalog)
    print(f"  {len(catalog)} items in catalog")

    v2 = classify_items(catalog)
    kept = sum(1 for v in v2.values() if v["kept"])
    print(f"v2 filter: kept {kept}/{len(v2)} items "
          f"(dropped {len(v2) - kept} with NO_CAUSAL_EFFECT_ON_SUFFIX)")

    # Breakdown by domain / intervention kind
    by_domain = defaultdict(lambda: [0, 0])  # [kept, total]
    by_kind = defaultdict(lambda: [0, 0])
    for iid, verdict in v2.items():
        it = catalog[iid]
        d = it["domain"]
        k = it["intervention"]["kind"]
        by_domain[d][1] += 1
        by_kind[k][1] += 1
        if verdict["kept"]:
            by_domain[d][0] += 1
            by_kind[k][0] += 1

    # json can't serialize tuple keys — flatten to string keys
    v2_serializable = {"::".join(map(str, k)): v for k, v in v2.items()}
    (args.out_dir / "v2_filter_verdicts.json").write_text(
        json.dumps(v2_serializable, indent=2))
    print(f"wrote {args.out_dir / 'v2_filter_verdicts.json'}")

    print("\nRejection rate by domain:")
    for d in sorted(by_domain):
        kept, total = by_domain[d]
        print(f"  {d:<12} kept {kept:3d}/{total:3d} "
              f"({(total - kept) / total * 100:.1f}% rejected)")
    print("\nRejection rate by intervention kind:")
    for k in sorted(by_kind):
        kept, total = by_kind[k]
        print(f"  {k:<20} kept {kept:3d}/{total:3d} "
              f"({(total - kept) / total * 100:.1f}% rejected)")

    # Re-aggregate each labeled result file
    agg = {}
    for rf in sorted(args.results_dir.glob("*_labeled.json")):
        name = rf.stem.replace("_labeled", "")
        print(f"\n{name}:")
        try:
            summary = aggregate_filtered(rf, v2, allow_mixed=args.allow_mixed)
        except rp.MixedRunError as e:
            raise SystemExit(f"error: {e}")
        except sp.InvalidRunError as e:
            raise SystemExit(f"error: {rf.name} does not match the catalog ({e}). Pass the catalog the "
                             "results were produced on with --catalog (e.g. counterfactuals/core_v1.jsonl).")
        agg[name] = {"file": rf.name, **summary}
        a = summary["summary_all_items"]
        b = summary["summary_after_v2_filter"]
        if a and b:
            print(f"  VM    all={a['vm']*100:.1f}%  v2-filtered={b['vm']*100:.1f}%")
            if a.get('expl_adj') is not None and b.get('expl_adj') is not None:
                print(f"  Expl  all={a['expl_adj']*100:.1f}%  v2-filtered={b['expl_adj']*100:.1f}%")
    # Also handle raw (non-labeled) JSONs (e.g. sonnet46_sliding.json); files that
    # are not result files are listed as skipped, provenance errors stop the run.
    skipped = []
    for rf in sorted(args.results_dir.glob("*.json")):
        if rf.stem.endswith("_labeled"):
            continue
        name = rf.stem
        if rf.parent.name.startswith("leaderboard"):
            continue
        if not is_result_file(rf):
            skipped.append({"file": rf.name, "reason": "not an evaluator result file (no per_item list)"})
            continue
        if name in agg:   # e.g. model.json next to model_labeled.json: keep both or stop
            raise SystemExit(f"error: {rf.name} and {agg[name]['file']} both report as {name!r}; rename one "
                             "of them or put them in separate --results-dir directories")
        try:
            summary = aggregate_filtered(rf, v2, allow_mixed=args.allow_mixed)
        except rp.MixedRunError as e:
            raise SystemExit(f"error: {e}")
        except sp.InvalidRunError as e:
            raise SystemExit(f"error: {rf.name} does not match the catalog ({e}). Pass the catalog the "
                             "results were produced on with --catalog (e.g. counterfactuals/core_v1.jsonl).")
        agg[name] = {"file": rf.name, **summary}
        a = summary["summary_all_items"]; b = summary["summary_after_v2_filter"]
        if a and b:
            print(f"\n{name} (raw):")
            print(f"  VM    all={a['vm']*100:.1f}%  v2-filtered={b['vm']*100:.1f}%")
            if a.get('expl_adj') is not None and b.get('expl_adj') is not None:
                print(f"  Expl  all={a['expl_adj']*100:.1f}%  v2-filtered={b['expl_adj']*100:.1f}%")

    if skipped:
        agg["_skipped_inputs"] = skipped
        print("\nskipped inputs:")
        for sk in skipped:
            print(f"  {sk['file']}: {sk['reason']}")
    (args.out_dir / "v2_filtered_leaderboard.json").write_text(
        json.dumps(agg, indent=2))
    print(f"\nwrote {args.out_dir / 'v2_filtered_leaderboard.json'}")


if __name__ == "__main__":
    main()
