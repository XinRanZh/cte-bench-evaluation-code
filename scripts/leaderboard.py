"""Aggregate per-model counterfactual-eval result JSONs into a leaderboard.

Reads N JSON files produced by `eval_llm_counterfactual.py`, emits:

  - <out_dir>/leaderboard.csv          : per (model, domain, kind) headline row
  - <out_dir>/leaderboard.md           : same, as a markdown table
  - <out_dir>/per_model_summary.md     : per-model one-liner summaries
  - <out_dir>/figures/value_match.png  : grouped bar chart by domain (top-level)
  - <out_dir>/figures/first_div.png    : histogram of first-divergence step
                                         per (model, domain)
  - <out_dir>/figures/kind_breakdown.png: per-intervention-kind headroom

matplotlib is optional: if not importable we emit CSV/MD only and print a
warning so the user can install it later.

Usage:
    python scripts/leaderboard.py \
        --result-jsons results/kimi.json results/sonnet46.json results/gpt41.json \
        --out-dir leaderboard/
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_provenance as rp  # noqa: E402
import grader  # noqa: E402


# ---------------------------------------------------------------------------
# Aggregation

PROVENANCE_REPORTS: list[dict] = []  # filled by aggregate(); written to provenance.json
GROUPS: list = []                    # the configuration groups of the last aggregate()


def _row_fields(stamp, desc) -> dict:
    src = stamp if isinstance(stamp, dict) else None
    return {k: (src.get(k) if src else desc["declared"].get(k)) for k in rp.REPORT_FIELDS}


DEFAULT_CATALOGS = sorted((Path(__file__).resolve().parent.parent / "counterfactuals").glob("*.jsonl"))


def _bind(path: Path, data: dict, desc: dict, catalogs: list[Path]) -> Path:
    """Bind one input to the catalog it was produced on; return that catalog.

    A stamped input binds to the catalog with its run_config's hash. Any other
    input binds to the first catalog whose rows it answers one-to-one with
    matching ids, calls and targets. An input that answers none raises."""
    import score_paired as sp
    cands = catalogs
    if desc["config"] is not None:
        cands = [c for c in catalogs if rp.file_sha256(c) == desc["config"].get("cf_jsonl_sha256")]
        if not cands:
            rp._mark(desc, "its rows were produced on none of the given catalogs")
            cands = catalogs
    errors = []
    for cat in cands:
        try:
            rp.bind_to_catalog(desc, data, sp.load_jsonl(cat), None, catalog_name=cat.name, require_complete=False)
            return cat
        except sp.InvalidRunError as e:
            errors.append(f"{cat.name}: {e}")
    raise SystemExit(f"error: leaderboard: {path} answers none of the catalogs "
                     f"({'; '.join(errors)}). Pass the catalog it was run on with --catalog.")


def config_groups(result_jsons: list[Path], allow_mixed: bool = False,
                  catalogs: list[Path] = ()) -> list[tuple[dict, list[dict]]]:
    """Group the rows of all inputs by their complete run configuration.

    Every input goes through run_provenance: damaged inputs are rejected; mixed
    inputs, inputs on a catalog other than the given --catalog files, and
    unstamped legacy inputs combined with stamped ones are rejected unless
    allow_mixed. Rows are pooled only under an identical configuration key
    (model, strategy, window, prompt version, thinking, effort, max_tokens,
    suffix budget, catalog); a legacy file is its own group. With allow_mixed a
    mixed input is split by row stamp and labelled MIXED. Every input is bound
    to its catalog (default: every catalog in counterfactuals/)."""
    catalogs = [Path(c) for c in catalogs] or DEFAULT_CATALOGS
    loaded = []
    for path in result_jsons:
        data = json.loads(path.read_text())
        desc = rp.describe_run(data, str(path))
        if desc["status"] != rp.DAMAGED:
            desc["bound_catalog"] = _bind(path, data, desc, catalogs)
        loaded.append((path, data, desc))
    descs = [d for _, _, d in loaded]
    prov = rp.check(descs, allow_mixed=allow_mixed, tool="leaderboard",
                    extra_problems=[p for p in rp.combination_problems(descs, fields=())
                                    if "legacy" in p])
    PROVENANCE_REPORTS.append(prov)
    GROUPS.clear()
    groups: dict = {}
    for path, data, desc in loaded:
        for r in data.get("per_item", []):
            if r is None:
                continue
            if desc["status"] == rp.LEGACY:
                fields = {**rp.report_fields(desc), "provenance": rp.LEGACY, "source_file": str(path),
                          "cf_jsonl_sha256": rp.file_sha256(desc["bound_catalog"]),
                          "catalog_name": desc["bound_catalog"].name}
                key = json.dumps(fields, sort_keys=True)
            else:
                fields = {**_row_fields(r.get(rp.ROW_CONFIG_KEY), desc),
                          "provenance": "MIXED" if desc["status"] == rp.MIXED else rp.STAMPED}
                key = json.dumps(fields, sort_keys=True)
            g = groups.setdefault(key, ({**fields, "source_files": []}, [], set()))
            if str(path) not in g[0]["source_files"]:
                g[0]["source_files"].append(str(path))
            ident = r.get(rp.ROW_ID_KEY) or (str(path), r.get("item_id"), id(r))
            if ident in g[2]:
                raise SystemExit(f"error: leaderboard: scenario {r.get('item_id')} appears twice under one "
                                 f"configuration ({', '.join(g[0]['source_files'])})")
            g[2].add(ident)
            g[1].append(r)
    labels: dict = {}
    for key, (f, rows, _) in groups.items():
        f["group_id"] = hashlib.sha256(key.encode()).hexdigest()[:12]
        lab = display_label(f)
        labels[lab] = labels.get(lab, 0) + 1
        f["label"] = lab if labels[lab] == 1 else f"{lab} #{labels[lab]}"
        GROUPS.append((f, rows))
    return list(GROUPS)


def display_label(fields: dict) -> str:
    if fields.get("label"):
        return fields["label"]
    if fields["provenance"] == rp.LEGACY:
        proto = f"{fields['strategy']} K={fields['window']}" if fields.get("window") is not None else str(fields["strategy"])
        return (f"{fields['model']} [{proto}] ({Path(fields['source_file']).name}, "
                f"catalog {fields.get('catalog_name')})")
    cat = (fields.get("cf_jsonl_sha256") or "")[:8]
    lab = (f"{fields['model']} [{fields['strategy']} K={fields['window']}, prompt {fields['prompt_version']}, "
           f"thinking {fields['thinking']}/{fields['reasoning_effort']}, max_tokens {fields['max_tokens']}, "
           f"budget {fields['suffix_budget']}, catalog {cat}]")
    return lab + (" (MIXED)" if fields["provenance"] == "MIXED" else "")


def aggregate(result_jsons: list[Path], allow_mixed: bool = False, catalogs: list[Path] = ()) -> list[dict]:
    """Return a list of dicts — one per (run configuration, domain, kind) — with:
       model (display label), model_id, strategy, window, prompt_version,
       thinking, reasoning_effort, max_tokens, catalog_sha256, provenance,
       domain, intervention_kind, n_items, suffix_total, value_match,
       status_match, well_formed, first_div_median, explain_exact_rate,
       explain_lineconv_rate, explain_adjacent_rate, state_match_rate.
    """
    PROVENANCE_REPORTS.clear()
    rows = []
    for fields, group_rows in config_groups(result_jsons, allow_mixed, catalogs):
        rows += _aggregate_group(fields, group_rows)
    return rows


def _aggregate_group(fields: dict, group_rows: list[dict]) -> list[dict]:
    rows = []
    model = display_label(fields)
    # Group per-item results by (domain, kind)
    bucket = defaultdict(list)
    for r in group_rows:
        if r is None or "error" in r:
            continue
        # calls that returned a provider error without model output are not scored
        bucket[(r["domain"], r["intervention_kind"])].append(grader.scoring_view(r))

    for (domain, kind), items in bucket.items():
        suffix_total = sum(i["suffix_len"] for i in items)
        vm = sum(i["n_value_match"] for i in items)
        sm = sum(i["n_status_match"] for i in items)
        wf = sum(i["n_well_formed"] for i in items)

        first_divs = [i["first_value_mismatch_at"] for i in items
                      if i.get("first_value_mismatch_at") is not None]

        # Phase B: explanation localization
        b_applicable = sum(1 for i in items
                           if i.get("phase_b") and i["phase_b"].get("applicable"))
        b_exact = sum(1 for i in items
                      if i.get("phase_b") and i["phase_b"].get("exact_match"))
        b_adj   = sum(1 for i in items
                      if i.get("phase_b") and i["phase_b"].get("adjacent_match"))
        b_conv = sum(_lineconv_hit(i) for i in items)

        # Phase C: state probe
        c_paths = sum(i["phase_c"]["n_paths"] for i in items
                      if i.get("phase_c") and i["phase_c"].get("valid"))
        c_match = sum(i["phase_c"]["n_match"] for i in items
                      if i.get("phase_c") and i["phase_c"].get("valid"))

        rows.append({
            "model": model,
            "group_id": fields["group_id"],
            "model_id": fields["model"],
            "strategy": fields["strategy"],
            "window": fields["window"],
            "prompt_version": fields["prompt_version"],
            "thinking": fields["thinking"],
            "reasoning_effort": fields["reasoning_effort"],
            "max_tokens": fields["max_tokens"],
            "catalog_sha256": fields.get("cf_jsonl_sha256"),
            "provenance": fields["provenance"],
            "domain": domain,
            "intervention_kind": kind,
            "n_items": len(items),
            "suffix_total": suffix_total,
            "value_match": vm,
            "status_match": sm,
            "well_formed": wf,
            "value_match_rate": (vm / suffix_total) if suffix_total else 0.0,
            "status_match_rate": (sm / suffix_total) if suffix_total else 0.0,
            "well_formed_rate": (wf / suffix_total) if suffix_total else 0.0,
            "first_div_median": (median(first_divs) if first_divs else None),
            "first_div_mean": (round(mean(first_divs), 2) if first_divs else None),
            "explain_applicable": b_applicable,
            "explain_exact_rate": (b_exact / b_applicable) if b_applicable else None,
            "explain_lineconv_rate": (b_conv / b_applicable) if b_applicable else None,
            "explain_adjacent_rate": (b_adj / b_applicable) if b_applicable else None,
            "state_paths": c_paths,
            "state_match": c_match,
            "state_match_rate": (c_match / c_paths) if c_paths else None,
            "infra_excluded_calls": sum(sum((i.get("n_infra_excluded") or {}).values()) for i in items),
        })
    return rows


def _lineconv_hit(item: dict) -> int:
    b = item.get("phase_b") or {}
    if not b.get("applicable"):
        return 0
    pred = b.get("extracted")
    if pred is None:
        pred = b.get("extracted_line")
    truth = b.get("truth_line")
    if pred is None or truth is None:
        return 0
    return int(int(pred) in (int(truth), int(truth) - 1))


# ---------------------------------------------------------------------------
# Output writers

CSV_COLUMNS = [
    "model", "group_id", "model_id", "strategy", "window", "prompt_version", "thinking", "reasoning_effort",
    "max_tokens", "catalog_sha256", "provenance",
    "domain", "intervention_kind",
    "n_items", "suffix_total",
    "value_match_rate", "status_match_rate", "well_formed_rate",
    "first_div_median", "first_div_mean",
    "explain_exact_rate", "explain_lineconv_rate", "explain_adjacent_rate",
    "state_match_rate", "infra_excluded_calls",
]


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k) for k in CSV_COLUMNS}
            # Round floats to 4 decimals for readability
            for k, v in list(row.items()):
                if isinstance(v, float):
                    row[k] = round(v, 4)
            w.writerow(row)


def _fmt(v, pct=False):
    if v is None: return "-"
    if isinstance(v, float):
        return f"{v*100:.1f}%" if pct else f"{v:.2f}"
    return str(v)


def _warning_lines() -> list[str]:
    return [f"> {rp.warning_line(p)}" for p in PROVENANCE_REPORTS if p["mixed"]]


def write_markdown(rows: list[dict], path: Path) -> None:
    # Sort rows for stable output
    rows_sorted = sorted(rows, key=lambda r: (r["model"], r["domain"], r["intervention_kind"]))
    warn = _warning_lines()
    lines = (warn + [""] if warn else []) + [
        "| model | domain | kind | n | value-match | status-match | first-div | expl. (exact/conv) | state-match |",
        "|---|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for r in rows_sorted:
        expl = (f"{_fmt(r['explain_exact_rate'], pct=True)} / "
                f"{_fmt(r['explain_lineconv_rate'], pct=True)}"
                if r["explain_applicable"] else "N/A")
        lines.append("| "
            f"{r['model']} | {r['domain']} | {r['intervention_kind']} | "
            f"{r['n_items']} | {_fmt(r['value_match_rate'], pct=True)} | "
            f"{_fmt(r['status_match_rate'], pct=True)} | "
            f"{_fmt(r['first_div_median'])} | "
            f"{expl} | "
            f"{_fmt(r['state_match_rate'], pct=True)} |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_per_model_summary(rows: list[dict], path: Path) -> None:
    by_model = defaultdict(list)    # one section per configuration group (never pooled)
    for r in rows:
        by_model[(r["model"], r["group_id"])].append(r)
    warn = _warning_lines()
    lines = ["# Per-model summary", ""] + (warn + [""] if warn else [])
    for key in sorted(by_model):
        model = key[0]
        items = by_model[key]
        total_suffix = sum(r["suffix_total"] for r in items)
        total_vm     = sum(r["value_match"] for r in items)
        total_sm     = sum(r["status_match"] for r in items)
        total_wf     = sum(r["well_formed"] for r in items)
        # Adversarial subset
        adv = [r for r in items if r["intervention_kind"] == "adversarial"]
        adv_suffix = sum(r["suffix_total"] for r in adv)
        adv_vm     = sum(r["value_match"] for r in adv)
        # Explanation
        expl_app = sum(r["explain_applicable"] for r in items)
        expl_conv = sum(r["explain_applicable"] * (r["explain_lineconv_rate"] or 0)
                        for r in items)
        lines += [
            f"## {model}",
            f"- overall value-match: **{_fmt(total_vm/total_suffix if total_suffix else 0.0, pct=True)}** "
                f"({total_vm}/{total_suffix} suffix steps)",
            f"- overall status-match: {_fmt(total_sm/total_suffix if total_suffix else 0.0, pct=True)}",
            f"- well-formed rate: {_fmt(total_wf/total_suffix if total_suffix else 0.0, pct=True)}",
            f"- adversarial subset value-match: "
                f"**{_fmt(adv_vm/adv_suffix if adv_suffix else 0.0, pct=True)}** "
                f"({adv_vm}/{adv_suffix})",
            (f"- explanation-localization line-convention match: "
                 f"**{_fmt(expl_conv/expl_app if expl_app else 0.0, pct=True)}** "
                 f"({int(round(expl_conv))}/{expl_app})"),
            "",
        ]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Plots (matplotlib optional)

def plot_value_match_by_domain(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable, skipping plots ({e})")
        return

    # Group by (domain, model) averaged across kinds weighted by suffix_total.
    grid = defaultdict(lambda: [0, 0])  # (domain, model) -> [vm, suffix]
    for r in rows:
        key = (r["domain"], r["model"])
        grid[key][0] += r["value_match"]
        grid[key][1] += r["suffix_total"]
    domains = sorted({d for d, _ in grid})
    models  = sorted({m for _, m in grid})
    if not domains or not models:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(domains) * 1.2), 4.5))
    bar_w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        ys = [(grid.get((d, m), [0, 0])[0] /
               grid.get((d, m), [0, 1])[1]
               if grid.get((d, m), [0, 0])[1] else 0)
              for d in domains]
        xs = [j + i * bar_w for j in range(len(domains))]
        ax.bar(xs, [y * 100 for y in ys], width=bar_w, label=m)
    ax.set_xticks([j + bar_w * (len(models) - 1) / 2 for j in range(len(domains))])
    ax.set_xticklabels(domains, rotation=15)
    ax.set_ylabel("value-match (%)")
    ax.set_title("CTE-Bench value-match by domain, averaged over intervention kinds")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "value_match.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_first_div_hist(rows: list[dict], out_dir: Path,
                        per_item_pool: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    by_model: dict[str, list[int]] = defaultdict(list)
    for item in per_item_pool:
        if item is None or "error" in item:
            continue
        fd = item.get("first_value_mismatch_at")
        if fd is not None:
            by_model[item["_model"]].append(fd)
    if not by_model:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    for m, vals in sorted(by_model.items()):
        ax.hist(vals, bins=20, alpha=0.5, label=f"{m} (n={len(vals)})",
                density=True)
    ax.set_xlabel("first divergence step in suffix")
    ax.set_ylabel("density")
    ax.set_title("Where counterfactual predictions first diverge from ground truth")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = out_dir / "first_div.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_kind_breakdown(rows: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    models = sorted({r["model"] for r in rows})
    kinds  = sorted({r["intervention_kind"] for r in rows})
    if not models or not kinds:
        return

    # grid[model][kind] = weighted value-match rate
    agg = {m: defaultdict(lambda: [0, 0]) for m in models}
    for r in rows:
        agg[r["model"]][r["intervention_kind"]][0] += r["value_match"]
        agg[r["model"]][r["intervention_kind"]][1] += r["suffix_total"]

    fig, ax = plt.subplots(figsize=(max(9, len(kinds) * 1.4), 4.5))
    bar_w = 0.8 / max(1, len(models))
    for i, m in enumerate(models):
        ys = [(agg[m][k][0] / agg[m][k][1] * 100 if agg[m][k][1] else 0)
              for k in kinds]
        xs = [j + i * bar_w for j in range(len(kinds))]
        ax.bar(xs, ys, width=bar_w, label=m)
    ax.set_xticks([j + bar_w * (len(models) - 1) / 2 for j in range(len(kinds))])
    ax.set_xticklabels(kinds, rotation=20)
    ax.set_ylabel("value-match (%)")
    ax.set_title("CTE-Bench headroom by intervention kind")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = out_dir / "kind_breakdown.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-jsons", nargs="+", required=True)
    ap.add_argument("--out-dir", default="leaderboard")
    ap.add_argument("--allow-mixed", action="store_true", help=rp.ALLOW_FLAG_HELP)
    ap.add_argument("--catalog", action="append", default=[], type=Path,
                    help="catalog JSONL the inputs must have been produced on (repeatable)")
    args = ap.parse_args()

    paths = [Path(p) for p in args.result_jsons]
    for p in paths:
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = aggregate(paths, allow_mixed=args.allow_mixed, catalogs=args.catalog)
    except rp.MixedRunError as e:
        raise SystemExit(f"error: {e}")
    for prov in PROVENANCE_REPORTS:
        if prov["mixed"]:
            print(rp.warning_line(prov))
    (out_dir / "provenance.json").write_text(json.dumps(
        {"mixed_run_provenance": any(p["mixed"] for p in PROVENANCE_REPORTS),
         "inputs": PROVENANCE_REPORTS}, indent=2) + "\n")
    write_csv(rows, out_dir / "leaderboard.csv")
    write_markdown(rows, out_dir / "leaderboard.md")
    write_per_model_summary(rows, out_dir / "per_model_summary.md")
    print(f"wrote {out_dir/'leaderboard.csv'} ({len(rows)} rows)")
    print(f"wrote {out_dir/'leaderboard.md'}")
    print(f"wrote {out_dir/'per_model_summary.md'}")

    # Per-item pool (tag by run-configuration label, for first-div histogram)
    pool: list[dict] = []
    for fields, group_rows in GROUPS:
        m = display_label(fields)
        for item in group_rows:
            if item is None or "error" in item:
                continue
            item2 = dict(item); item2["_model"] = m
            pool.append(item2)

    plot_value_match_by_domain(rows, fig_dir)
    plot_first_div_hist(rows, fig_dir, pool)
    plot_kind_breakdown(rows, fig_dir)


if __name__ == "__main__":
    main()
