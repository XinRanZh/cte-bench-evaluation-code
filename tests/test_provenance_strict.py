"""Strict provenance validation: one regression test per reproduction of review round 8.

All offline: no model is called. Fixtures are written directly in the
evaluator's output format (stamped rows carry run_config and
catalog_row_sha256; legacy rows carry neither).
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import apply_v2_filter as av2  # noqa: E402
import audit_reference_results as ar  # noqa: E402
import build_core_v1 as bcv1  # noqa: E402
import eval_llm_counterfactual as E  # noqa: E402
import leaderboard as lb  # noqa: E402
import run_provenance as rp  # noqa: E402
import run_state_probe_v2 as rsp  # noqa: E402
import score_paired as sp  # noqa: E402

BUDGET = 40
CORE_PATH = ROOT / "counterfactuals" / "core_v1.jsonl"
NE_PATH = ROOT / "counterfactuals" / "no_effect_v1.jsonl"
ADV_PATH = ROOT / "counterfactuals" / "adv24_reasoning_probe.jsonl"
REN_PATH = ROOT / "counterfactuals" / "core_v1_renamed.jsonl"
CORE = sp.load_jsonl(CORE_PATH)
NE = sp.load_jsonl(NE_PATH)
REF = os.environ.get("CTE_REFERENCE_RESULTS")


def config(cf, **over):
    cfg = {"model": "A", "strategy": "sliding", "window": 20, "prompt_version": "v1", "suffix_budget": BUDGET,
           "thinking": "off", "reasoning_effort": "high", "max_tokens": 4000, "explain": False, "state": False,
           "cf_jsonl_sha256": rp.file_sha256(cf)}
    cfg.update(over)
    return cfg


def rows_for(catalog, stamp=None, correct=True):
    out = []
    for r in catalog:
        steps = []
        for j, s in enumerate(r["trace_counterfactual"][:BUDGET]):
            pred = s["response"] if correct else {"error": {"code": "WRONG"}}
            steps.append({"suffix_idx": j, "op": s["op"], "args": s["args"], "predicted": pred,
                          "ground_truth": s["response"], "value_match": correct, "status_match": correct,
                          "well_formed": True})
        n = len(steps)
        row = {"item_id": r["item_id"], "domain": r["domain"], "intervention_kind": r["intervention"]["kind"],
               "prefix_len": r["prefix_len"], "suffix_len": n, "n_well_formed": n,
               "n_value_match": n if correct else 0, "n_status_match": n if correct else 0,
               "first_value_mismatch_at": None if correct else 0, "expected_delay": None,
               "billing": {"n_calls": n}, "phase_a": steps, "phase_b": None, "phase_c": None}
        if stamp is not None:
            row[rp.ROW_CONFIG_KEY] = stamp
            row[rp.ROW_ID_KEY] = rp.catalog_row_id(r)
        out.append(row)
    return out


def write_output(path, catalog, stamp=None, correct=True, declared=None):
    """An evaluator output: stamped if `stamp` is given, else unstamped legacy."""
    rows = rows_for(catalog, stamp, correct)
    if stamp is None:
        top = {"model": "A", "strategy": "sliding", "window": 20, "thinking": "off", "suffix_budget": BUDGET}
    else:
        top = {k: stamp[k] for k in ("model", "strategy", "window", "prompt_version", "thinking",
                                     "reasoning_effort", "max_tokens", "suffix_budget")}
        top.update({"run_config": stamp, "mixed_run_configurations": False})
    top.update(declared or {})
    top["per_item"] = rows
    Path(path).write_text(json.dumps(top))
    return Path(path)


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp_obj = tempfile.TemporaryDirectory()
        self.tmp = Path(self.tmp_obj.name)

    def tearDown(self):
        self.tmp_obj.cleanup()


class TestLeaderboardConfigurations(Base):
    """Separate clean files with different settings are never pooled."""

    def test_different_settings_are_separate_groups(self):
        a = write_output(self.tmp / "a.json", CORE, config(CORE_PATH, window=20, thinking="off", max_tokens=4000))
        b = write_output(self.tmp / "b.json", CORE, config(CORE_PATH, window=7, thinking="on", max_tokens=12000),
                         correct=False)
        rows = lb.aggregate([a, b])
        labels = {r["model"] for r in rows}
        self.assertEqual(len(labels), 2)
        self.assertEqual({(r["window"], r["thinking"], r["max_tokens"]) for r in rows},
                         {(20, "off", 4000), (7, "on", 12000)})
        out_dir = self.tmp / "lb"
        with mock.patch.object(sys, "argv", ["lb", "--result-jsons", str(a), str(b), "--out-dir", str(out_dir)]):
            quiet(lb.main)
        summary = (out_dir / "per_model_summary.md").read_text()
        self.assertIn("**100.0%**", summary)
        self.assertIn("**0.0%**", summary)
        self.assertNotIn("**50.0%**", summary)

    def test_legacy_and_stamped_files_are_not_pooled(self):
        a = write_output(self.tmp / "a.json", CORE, config(CORE_PATH))
        legacy = write_output(self.tmp / "legacy.json", CORE)
        with self.assertRaises(rp.MixedRunError):
            lb.aggregate([a, legacy])
        rows = lb.aggregate([a, legacy], allow_mixed=True)
        self.assertEqual({r["provenance"] for r in rows}, {rp.STAMPED, rp.LEGACY})
        self.assertTrue(lb.PROVENANCE_REPORTS[0]["mixed"])

    def test_same_configuration_split_across_files_is_pooled_once(self):
        cfg = config(CORE_PATH)
        a = write_output(self.tmp / "a.json", CORE[:100], cfg)
        b = write_output(self.tmp / "b.json", CORE[100:], cfg)
        rows = lb.aggregate([a, b])
        self.assertEqual(sum(r["n_items"] for r in rows), 255)
        c = write_output(self.tmp / "c.json", CORE[50:120], cfg)
        with self.assertRaises(SystemExit):
            lb.aggregate([a, c])                       # overlapping scenarios


class TestDeclarationsAndDamage(Base):
    def test_declaration_contradicting_stamps_is_rejected_and_stamps_win(self):
        core = write_output(self.tmp / "core.json", CORE, config(CORE_PATH), declared={"thinking": "on"})
        ne = write_output(self.tmp / "ne.json", NE, config(NE_PATH), declared={"thinking": "on"})
        with self.assertRaises(SystemExit) as cm:
            quiet(sp.main, ["--core-run", str(core), "--no-effect-run", str(ne), "--label", "A"])
        self.assertIn("declares thinking='on'", str(cm.exception))
        out = self.tmp / "p.json"
        quiet(sp.main, ["--core-run", str(core), "--no-effect-run", str(ne), "--label", "A", "--allow-mixed",
                        "--json", str(out)])
        m = json.loads(out.read_text())["models"]["A"]
        self.assertEqual(m["core_run"]["thinking"], "off")
        self.assertTrue(m["mixed_run_provenance"])

    def test_top_level_run_config_contradicting_stamps_is_rejected(self):
        core = write_output(self.tmp / "core.json", CORE, config(CORE_PATH),
                            declared={"run_config": config(CORE_PATH, model="B")})
        with self.assertRaises(rp.MixedRunError):
            lb.aggregate([core])

    def test_damaged_output_is_rejected_even_with_allow_mixed(self):
        core = write_output(self.tmp / "core.json", CORE, config(CORE_PATH), declared={"model": "B"})
        data = json.loads(core.read_text())
        for r in data["per_item"]:
            del r[rp.ROW_CONFIG_KEY]
        core.write_text(json.dumps(data))
        ne = write_output(self.tmp / "ne.json", NE, config(NE_PATH))
        for fn in (lambda: lb.aggregate([core], allow_mixed=True),
                   lambda: quiet(sp.main, ["--core-run", str(core), "--no-effect-run", str(ne), "--label", "B",
                                           "--allow-mixed"])):
            with self.assertRaises((rp.MixedRunError, SystemExit)) as cm:
                fn()
            self.assertIn("damaged", str(cm.exception))


class TestCatalogBinding(Base):
    def test_wrong_catalog_hash_is_rejected_by_audit_filter_and_core_builder(self):
        stamp = config(CORE_PATH, cf_jsonl_sha256="0" * 64)
        path = write_output(self.tmp / "ds_lite_sliding_labeled.json", CORE,
                            {**stamp, "model": "deepseek/deepseek-chat"})
        with self.assertRaises(rp.MixedRunError):
            ar.audit_file(path, CORE, "schedule", catalog_path=CORE_PATH)
        av2.CATALOG_SHAS.clear(); av2.CATALOG_SHAS.add(rp.file_sha256(CORE_PATH))
        try:
            with self.assertRaises(rp.MixedRunError):
                av2.aggregate_filtered(path, {})
        finally:
            av2.CATALOG_SHAS.clear()
        bcv1.CORE_CATALOG_SHA = rp.file_sha256(CORE_PATH)
        try:
            with self.assertRaises(rp.MixedRunError):
                quiet(bcv1.load_result_items, self.tmp, path.name)
        finally:
            bcv1.CORE_CATALOG_SHA = None

    def test_core_builder_rejects_runs_that_are_not_the_reference_runs(self):
        path = write_output(self.tmp / "ds_lite_sliding_labeled.json", CORE, config(CORE_PATH, model="A"))
        with self.assertRaises(rp.MixedRunError) as cm:
            quiet(bcv1.load_result_items, self.tmp, path.name)
        self.assertIn("expects model", str(cm.exception))


class TestRenamedStudy(Base):
    def setUp(self):
        super().setUp()
        (self.tmp / "counterfactuals").mkdir()
        for p in (CORE_PATH, REN_PATH):
            (self.tmp / "counterfactuals" / p.name).symlink_to(p)
        self.ren = sp.load_jsonl(REN_PATH)
        ids = {r["paired_item_id"] for r in self.ren}
        self.orig = [r for r in CORE if r["item_id"] in ids]

    def _arms(self, **cfg_over):
        d = self.tmp / "reference_results" / "renamed_raw"
        d.mkdir(parents=True, exist_ok=True)
        for arm in ("original_run1", "original_run2", "renamed_run1", "renamed_run2"):
            renamed = arm.startswith("renamed")
            cat, cf = (self.ren, REN_PATH) if renamed else (self.orig, CORE_PATH)
            write_output(d / f"m_{arm}_tf1step_sliding_k20.json", cat, config(cf, **cfg_over))

    def _run(self, *extra):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "score_renamed_variant.py"), "--dataset-root",
                               str(self.tmp), "--json", str(self.tmp / "out.json"), *extra],
                              capture_output=True, text=True)

    def test_clean_study_reports_settings_from_stamps(self):
        self._arms()
        r = self._run()
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads((self.tmp / "out.json").read_text())
        self.assertEqual((d["protocol"], d["model"]), ("tf1step_sliding_k20", "A"))

    def test_free_rollout_k7_arms_are_rejected(self):
        self._arms(strategy="free_rollout", window=7)
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expects strategy", r.stderr)

    def test_original_outputs_in_the_renamed_arms_are_rejected(self):
        d = self.tmp / "reference_results" / "renamed_raw"
        d.mkdir(parents=True)
        for arm in ("original_run1", "original_run2", "renamed_run1", "renamed_run2"):
            write_output(d / f"m_{arm}_tf1step_sliding_k20.json", self.orig)   # legacy, original rows everywhere
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid arm", r.stderr)

    def test_malformed_step_is_rejected(self):
        self._arms()
        p = self.tmp / "reference_results" / "renamed_raw" / "m_original_run1_tf1step_sliding_k20.json"
        data = json.loads(p.read_text())
        data["per_item"][0]["phase_a"].append({**data["per_item"][0]["phase_a"][4], "suffix_idx": -36,
                                               "op": "foreign", "predicted": {"x": 1}})
        p.write_text(json.dumps(data))
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("invalid suffix_idx -36", r.stderr)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the HF reference_results/ directory")
    def test_malformed_step_in_the_released_file_is_rejected(self):
        src = Path(REF) / "renamed_raw"
        dst = self.tmp / "reference_results" / "renamed_raw"
        shutil.copytree(src, dst)
        p = next(dst.glob("*_original_run1_*.json"))
        data = json.loads(p.read_text())
        row = data["per_item"][0]
        row["phase_a"].append({**row["phase_a"][4], "suffix_idx": -36, "op": "foreign_call"})
        p.write_text(json.dumps(data))
        r = self._run()
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("-36", r.stderr)


class TestReasoningStudy(Base):
    def test_outputs_of_another_model_and_protocol_are_rejected(self):
        (self.tmp / "counterfactuals").mkdir()
        (self.tmp / "counterfactuals" / CORE_PATH.name).symlink_to(CORE_PATH)
        for proto in ("tf1step_sliding_k20", "no_suffix_feedback"):
            for sub, name, thinking in (("raw", f"deepseek_v4_flash_{proto}.json", "off"),
                                        ("reasoning_raw", f"deepseek_v4_flash_thinking_{proto}.json", "on")):
                (self.tmp / "reference_results" / sub).mkdir(parents=True, exist_ok=True)
                write_output(self.tmp / "reference_results" / sub / name, CORE,
                             config(CORE_PATH, model="A", strategy="free_rollout", thinking=thinking))
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "score_reasoning_pairs.py"), "--dataset-root",
                            str(self.tmp), "--json", str(self.tmp / "out.json")], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("expects strategy", r.stderr)
        self.assertIn("expects model", r.stderr)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "score_reasoning_pairs.py"), "--dataset-root",
                            str(self.tmp), "--json", str(self.tmp / "out.json"), "--allow-mixed"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads((self.tmp / "out.json").read_text())
        self.assertTrue(d["model"].startswith("MIXED"))
        self.assertEqual(d["protocols"]["tf1step_sliding_k20"]["on"]["strategy"], "free_rollout")
        self.assertEqual(d["protocols"]["tf1step_sliding_k20"]["on"]["model"], "A")


class TestApplyV2Filter(Base):
    def _run(self, results):
        cf = self.tmp / "cf"
        cf.mkdir()
        (cf / ADV_PATH.name).symlink_to(ADV_PATH)
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "apply_v2_filter.py"), "--cf-dir", str(cf),
                               "--results-dir", str(results), "--out-dir", str(self.tmp / "out")],
                              capture_output=True, text=True)

    def test_mixed_raw_file_stops_the_run(self):
        res = self.tmp / "results"
        res.mkdir()
        adv = sp.load_jsonl(ADV_PATH)
        p = write_output(res / "model.json", adv, config(ADV_PATH))
        data = json.loads(p.read_text())
        data["mixed_run_configurations"] = True
        p.write_text(json.dumps(data))
        r = self._run(res)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("mixed run provenance", r.stderr)

    def test_non_result_files_are_listed_as_skipped(self):
        res = self.tmp / "results"
        res.mkdir()
        write_output(res / "model.json", sp.load_jsonl(ADV_PATH), config(ADV_PATH))
        (res / "notes.json").write_text(json.dumps({"hello": 1}))
        r = self._run(res)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads((self.tmp / "out" / "v2_filtered_leaderboard.json").read_text())
        self.assertIn("model", out)
        self.assertEqual(out["_skipped_inputs"], [{"file": "notes.json",
                                                   "reason": "not an evaluator result file (no per_item list)"}])


class TestStateProbeResume(Base):
    """A/off then B/on with the same calls directory must not reuse A's records."""

    def _run(self, model, thinking, *extra):
        argv = ["run_state_probe_v2.py", "--cf-jsonl", str(CORE_PATH), "--model", model, "--thinking", thinking,
                "--max-items", "2", "--max-concurrency", "1", "--calls-dir", str(self.tmp / "calls"),
                "--out", str(self.tmp / f"out_{model}_{thinking}.json"), *extra]
        calls = []

        def fake_call(model_id, *_a, **_k):
            calls.append(model_id)
            return "{}", {"prompt_tokens": 1}, 0.0
        with mock.patch.object(sys, "argv", argv), mock.patch.object(rsp, "call_llm", fake_call), \
                contextlib.redirect_stdout(io.StringIO()):
            rsp.main()
        return calls, json.loads((self.tmp / f"out_{model}_{thinking}.json").read_text())

    def test_other_configuration_is_rejected(self):
        calls, out = self._run("A", "off")
        self.assertEqual(len(calls), 2)
        self.assertFalse(out["mixed_run_configurations"])
        with self.assertRaises(SystemExit) as cm:
            self._run("B", "on")
        self.assertIn("not produced under this run's configuration", str(cm.exception))
        calls, out = self._run("B", "on", "--allow-mixed")
        self.assertEqual(calls, [])
        self.assertTrue(out["mixed_run_configurations"])
        self.assertEqual({c["run_config"]["model"] for c in out["row_run_configurations"]}, {"A"})

    def test_same_configuration_resumes_cleanly(self):
        self._run("A", "off")
        calls, out = self._run("A", "off")
        self.assertEqual(calls, [])
        self.assertFalse(out["mixed_run_configurations"])
        self.assertTrue(all(r["run_config"]["model"] == "A" and r[rp.ROW_ID_KEY] for r in out["records"]))


class TestReferencePolicyCatalog(Base):
    def test_other_catalog_needs_allow_mixed(self):
        cat = self.tmp / "other.jsonl"
        cat.write_text("\n".join(CORE_PATH.read_text().splitlines()[:5]) + "\n")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "reference_policies.py"), "--core-jsonl", str(cat),
                            "--bootstrap", "10"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not the Core-v1 catalog", r.stderr)


if __name__ == "__main__":
    unittest.main()
