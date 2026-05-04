"""Post-hoc application of reachability filter v2 and state-probe v2.

No model calls. Takes (a) the item catalog under `counterfactuals/*.jsonl`
and (b) existing labeled result JSONs, and produces:

  - A per-item filter verdict file with n_diff, first_diff_step, and
    (for state items) reachability-v2 reason.
  - A re-aggregated leaderboard where items that v2 rejects are removed.

Usage:
  python scripts/apply_v2_filter.py \\
      --cf-dir counterfactuals \\
      --results-dir results/full_grid_2026-04-29 \\
      --out-dir results/full_grid_2026-04-29/v2_filter
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reachability import check_suffix_causal_effect


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


def load_catalog(cf_dir: Path) -> dict[tuple, dict]:
    """Load counterfactual catalog keyed by globally unique row tuple."""
    out = {}
    row_index = 0
    for jf in sorted(cf_dir.glob("*.jsonl")):
        if jf.name == "core_v1.jsonl":
            continue
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


def aggregate_filtered(result_path: Path, v2_filter: dict[tuple, dict]) -> dict:
    """Recompute per-model totals, excluding rows v2 rejects.

    Result JSONs preserve catalog order, so row index is the primary
    alignment key. We still check the (item_id, domain, kind) triplet and
    fall back to a triplet lookup for partial result files.
    """
    d = json.loads(result_path.read_text())
    per_item = [r for r in d.get("per_item", []) if r and "error" not in r]

    ordered_keys = sorted(v2_filter.keys(), key=lambda k: k[0])
    by_triplet = defaultdict(list)
    for k in v2_filter.keys():
        item_id, domain, kind = k[1], k[2], k[3]
        by_triplet[(item_id, domain, kind)].append(k)

    kept = []
    dropped = []
    for idx, r in enumerate(per_item):
        triplet = (r.get("item_id"), r.get("domain"), r.get("intervention_kind"))
        verdict = None
        if idx < len(ordered_keys):
            key = ordered_keys[idx]
            if (key[1], key[2], key[3]) == triplet:
                verdict = v2_filter.get(key, {})
        if verdict is None:
            candidates = by_triplet.get(triplet, [])
            if not candidates:
                dropped.append(r)
                continue
            if len(candidates) == 1:
                verdict = v2_filter.get(candidates[0], {})
            else:
                # Partial runs may include a repeated item_id/domain/kind
                # cluster without enough fields to identify the exact row.
                # Keep the conservative majority fallback for those cases.
                n_keep = sum(1 for c in candidates
                             if v2_filter.get(c, {}).get("kept"))
                verdict = {"kept": n_keep >= len(candidates) / 2,
                           "reason": "MAJORITY_VOTE",
                           "n_candidates": len(candidates),
                           "n_keep_candidates": n_keep}
        if verdict.get("kept"):
            kept.append(r)
        else:
            dropped.append(r)

    def summarize(items):
        if not items:
            return None
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
        }

    return {
        "model": d.get("model"),
        "n_items_before_v2": len(per_item),
        "n_items_after_v2": len(kept),
        "n_items_dropped_by_v2": len(dropped),
        "summary_all_items": summarize(per_item),
        "summary_after_v2_filter": summarize(kept),
        "summary_of_dropped_items": summarize(dropped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-dir", required=True, type=Path,
                    help="directory of counterfactuals/*.jsonl")
    ap.add_argument("--results-dir", required=True, type=Path,
                    help="directory of per-model result JSONs")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"loading item catalog from {args.cf_dir} ...")
    catalog = load_catalog(args.cf_dir)
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
        summary = aggregate_filtered(rf, v2)
        agg[name] = summary
        a = summary["summary_all_items"]
        b = summary["summary_after_v2_filter"]
        if a and b:
            print(f"  VM    all={a['vm']*100:.1f}%  v2-filtered={b['vm']*100:.1f}%")
            if a.get('expl_adj') is not None and b.get('expl_adj') is not None:
                print(f"  Expl  all={a['expl_adj']*100:.1f}%  v2-filtered={b['expl_adj']*100:.1f}%")
    # Also handle raw (non-labeled) JSONs (e.g. sonnet46_sliding.json)
    for rf in sorted(args.results_dir.glob("*.json")):
        if rf.stem.endswith("_labeled"):
            continue
        name = rf.stem
        if name in agg:
            continue
        if rf.parent.name.startswith("leaderboard"):
            continue
        try:
            summary = aggregate_filtered(rf, v2)
        except Exception:
            continue
        agg[name] = summary
        a = summary["summary_all_items"]; b = summary["summary_after_v2_filter"]
        if a and b:
            print(f"\n{name} (raw):")
            print(f"  VM    all={a['vm']*100:.1f}%  v2-filtered={b['vm']*100:.1f}%")
            if a.get('expl_adj') is not None and b.get('expl_adj') is not None:
                print(f"  Expl  all={a['expl_adj']*100:.1f}%  v2-filtered={b['expl_adj']*100:.1f}%")

    (args.out_dir / "v2_filtered_leaderboard.json").write_text(
        json.dumps(agg, indent=2))
    print(f"\nwrote {args.out_dir / 'v2_filtered_leaderboard.json'}")


if __name__ == "__main__":
    main()
