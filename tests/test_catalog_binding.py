"""Catalog binding and no pooling: one regression test per reproduction of review round 9.

N1: the leaderboard pooled unstamped legacy files of one model (the three
    released Qwen outputs gave one 48.1% summary).
N2: apply_v2_filter with the development pool and Core results reported
    255 -> 0 retained instead of stopping.
N3: the leaderboard accepted a legacy Core result with --catalog no_effect_v1.

All offline: no model is called. Tests marked REF also run on the released
reference outputs when CTE_REFERENCE_RESULTS points at reference_results/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_provenance_strict import (  # noqa: E402
    BUDGET, CORE, CORE_PATH, NE_PATH, REF, ROOT, Base, quiet, rows_for, write_output)

import build_core_v1 as bcv1  # noqa: E402
import core_v1  # noqa: E402
import leaderboard as lb  # noqa: E402
import run_provenance as rp  # noqa: E402
import score_paired as sp  # noqa: E402

QWEN = ("qwen36_35b_a3b_free_rollout.json", "qwen36_35b_a3b_no_suffix_feedback.json",
        "qwen36_35b_a3b_tf1step_sliding_k20.json")


def legacy_output(path, catalog, correct_rows, declared=None):
    """An unstamped legacy output whose first `correct_rows` rows are right."""
    rows = rows_for(catalog[:correct_rows]) + rows_for(catalog[correct_rows:], correct=False)
    top = {"model": "A", "strategy": "sliding", "window": 20, "thinking": "off", "suffix_budget": BUDGET}
    top.update(declared or {})
    top["per_item"] = rows
    Path(path).write_text(json.dumps(top))
    return Path(path)


def run_leaderboard(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "leaderboard.py"), *map(str, args)],
                          capture_output=True, text=True)


def run_v2(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "apply_v2_filter.py"), *map(str, args)],
                          capture_output=True, text=True)


class TestLegacyFilesAreNeverPooled(Base):
    """N1: each legacy file is its own row, section and plot entry."""

    def test_three_legacy_files_of_one_model_stay_separate(self):
        files = [legacy_output(self.tmp / f"{n}.json", CORE, k, {"strategy": s})
                 for n, k, s in (("all", 255, "sliding"), ("none", 0, "free_rollout"), ("some", 51, "prefix_only"))]
        # even identical declarations do not pool two legacy files
        files.append(legacy_output(self.tmp / "twin.json", CORE, 0, {"strategy": "sliding"}))
        out_dir = self.tmp / "lb"
        with mock.patch.object(sys, "argv", ["lb", "--result-jsons", *map(str, files), "--out-dir", str(out_dir)]):
            quiet(lb.main)
        rows = lb.aggregate(files)
        self.assertEqual(len({r["group_id"] for r in rows}), 4)
        self.assertEqual({r["model_id"] for r in rows}, {"A"})
        summary = (out_dir / "per_model_summary.md").read_text()
        self.assertEqual(summary.count("## A ["), 4)
        for pct in ("**100.0%**", "**0.0%**", "**20.0%**"):
            self.assertIn(pct, summary)
        for pooled in ("**50.0%**", "**40.0%**", "**30.0%**"):   # any pooling of the four files
            self.assertNotIn(pooled, summary)
        for f in files:
            self.assertIn(f"({f.name}, catalog core_v1.jsonl)", summary)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the released reference_results/")
    def test_released_qwen_outputs_are_three_rows(self):
        paths = [Path(REF) / "raw" / f for f in QWEN]
        out_dir = self.tmp / "lb_qwen"
        r = run_leaderboard("--result-jsons", *paths, "--out-dir", out_dir)
        self.assertEqual(r.returncode, 0, r.stderr)
        summary = (out_dir / "per_model_summary.md").read_text()
        self.assertEqual(summary.count("## openrouter/qwen/qwen3.6-35b-a3b ["), 3)
        for pct in ("**65.6%** (6687/10200", "**34.9%** (3557/10200", "**43.8%** (4463/10200"):
            self.assertIn(pct, summary)
        self.assertNotIn("48.1%", summary)
        rows = lb.aggregate(paths)
        self.assertEqual(len({r["group_id"] for r in rows}), 3)


class TestWrongCatalogIsRejected(Base):
    """N3: a legacy result is checked against the catalog it is scored on."""

    def test_legacy_core_result_with_the_no_effect_catalog(self):
        path = legacy_output(self.tmp / "core_legacy.json", CORE, 255)
        with self.assertRaises(SystemExit) as cm:
            lb.aggregate([path], catalogs=[NE_PATH])
        self.assertIn("answers none of the catalogs", str(cm.exception))
        r = run_leaderboard("--result-jsons", path, "--catalog", NE_PATH, "--out-dir", self.tmp / "lb")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("answers none of the catalogs", r.stderr)
        self.assertEqual(sum(x["n_items"] for x in lb.aggregate([path], catalogs=[CORE_PATH])), 255)

    def test_legacy_result_that_answers_no_catalog(self):
        path = legacy_output(self.tmp / "tampered.json", CORE, 255)
        data = json.loads(path.read_text())
        data["per_item"][7]["phase_a"][2]["ground_truth"] = {"tampered": True}
        path.write_text(json.dumps(data))
        with self.assertRaises(SystemExit):
            lb.aggregate([path])

    def test_filter_core_results_binds_to_the_core_catalog(self):
        rows = rows_for(CORE)
        self.assertEqual(len(core_v1.filter_core_results(rows)), 255)
        rows[3]["phase_a"][0]["ground_truth"] = {"tampered": True}
        with self.assertRaises(sp.InvalidRunError):
            core_v1.filter_core_results(rows)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the released reference_results/")
    def test_released_qwen_core_result_with_the_no_effect_catalog(self):
        path = Path(REF) / "raw" / QWEN[2]
        r = run_leaderboard("--result-jsons", path, "--catalog", NE_PATH, "--out-dir", self.tmp / "lb")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("answers none of the catalogs", r.stderr)


class TestApplyV2FilterBindsTheCatalog(Base):
    """N2: rows are joined to the selected catalog; rows outside it stop the run."""

    def _results(self):
        res = self.tmp / "results"
        res.mkdir()
        legacy_output(res / "model.json", CORE, 255)
        return res

    def test_core_catalog_keeps_every_core_row(self):
        res = self._results()
        r = run_v2("--catalog", CORE_PATH, "--results-dir", res, "--out-dir", self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads((self.tmp / "out" / "v2_filtered_leaderboard.json").read_text())
        self.assertEqual((out["model"]["n_items_before_v2"], out["model"]["n_items_after_v2"]), (255, 255))

    def test_development_pool_with_core_results_stops(self):
        res = self._results()
        r = run_v2("--cf-dir", ROOT / "counterfactuals", "--results-dir", res, "--out-dir", self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not match the catalog", r.stderr)
        self.assertFalse((self.tmp / "out" / "v2_filtered_leaderboard.json").exists())

    def test_a_catalog_is_required(self):
        r = run_v2("--results-dir", self._results(), "--out-dir", self.tmp / "out")
        self.assertNotEqual(r.returncode, 0)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the released reference_results/")
    def test_released_outputs(self):
        raw = Path(REF) / "raw"
        r = run_v2("--catalog", CORE_PATH, "--results-dir", raw, "--out-dir", self.tmp / "out")
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads((self.tmp / "out" / "v2_filtered_leaderboard.json").read_text())
        runs = {k: v for k, v in out.items() if not k.startswith("_")}
        self.assertEqual(len(runs), 12)
        self.assertEqual({(v["n_items_before_v2"], v["n_items_after_v2"]) for v in runs.values()}, {(255, 255)})
        r = run_v2("--cf-dir", ROOT / "counterfactuals", "--results-dir", raw, "--out-dir", self.tmp / "out2")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("does not match the catalog", r.stderr)


class TestCoreBuilderBindsThePool(Base):
    """build_core_v1 joins the reference runs to the candidate pool."""

    def setUp(self):
        super().setUp()
        a, b = CORE[0], CORE[1]
        # a pool with a repeated id: the second row is labelled '<id>__2' in the runs
        self.pool = [a, {**b, "item_id": a["item_id"]}]
        bcv1.POOL_ROWS[:] = self.pool

    def tearDown(self):
        bcv1.POOL_ROWS.clear()
        super().tearDown()

    def _output(self, rows):
        path = self.tmp / "ds_lite_sliding_labeled.json"
        path.write_text(json.dumps({"model": "DS-V4-Lite/sliding", "strategy": "sliding", "window": 20,
                                    "thinking": "off", "suffix_budget": BUDGET, "per_item": rows}))
        return path

    def test_repeated_pool_id_is_joined_by_content(self):
        rows = rows_for(self.pool)
        rows[1]["item_id"] = f"{self.pool[0]['item_id']}__2"
        got = quiet(bcv1.load_result_items, self.tmp, self._output(rows).name)
        self.assertEqual([r["item_id"] for r in got], [r["item_id"] for r in rows])

    def test_row_outside_the_pool_stops(self):
        rows = rows_for(self.pool)
        rows[1]["phase_a"][0]["ground_truth"] = {"tampered": True}
        with self.assertRaises(SystemExit) as cm:
            quiet(bcv1.load_result_items, self.tmp, self._output(rows).name)
        self.assertIn("does not match the candidate pool", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
