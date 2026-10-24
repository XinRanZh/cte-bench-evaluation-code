"""Infrastructure failures and input validation: one regression test per
reproduction of review round 10 (F1-F10) and for the exclusion rule.

Exclusion rule: a call that returned a provider error without any model output
is excluded from every metric's numerator and denominator
(grader.is_infrastructure_failure / grader.scoring_view). Billed empty
responses, refusals and truncations are model outputs and stay scored.

All offline: no model is called (the evaluator and the state probe run with
in-process stand-ins). Tests marked REF also run on the released reference
outputs when CTE_REFERENCE_RESULTS points at reference_results/.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_provenance_strict import (  # noqa: E402
    BUDGET, CORE, CORE_PATH, NE, NE_PATH, REF, ROOT, Base, config, quiet, rows_for, write_output)

import apply_v2_filter as av2  # noqa: E402,F401
import audit_reference_results as ar  # noqa: E402
import build_core_v1 as bcv1  # noqa: E402
import eval_llm_counterfactual as E  # noqa: E402
import grader  # noqa: E402
import leaderboard as lb  # noqa: E402
import run_provenance as rp  # noqa: E402
import run_state_probe_v2 as rsp  # noqa: E402
import score_paired as sp  # noqa: E402

ERR = "ERR:litellm.ServiceUnavailableError: BedrockException - The system encountered an unexpected error"


def fail_steps(rows, picks):
    """Turn (row, step) positions into provider errors, as the evaluator stored them."""
    for i, j in picks:
        s = rows[i]["phase_a"][j]
        s.update({"predicted": None, "value_match": False, "status_match": False, "well_formed": False,
                  "raw_excerpt": ERR})
        rows[i]["n_value_match"] -= 1
        rows[i]["n_status_match"] -= 1
        rows[i]["n_well_formed"] -= 1
        rows[i]["billing"]["n_calls"] -= 1    # the evaluator bills only calls that returned output
    return rows


def legacy(path, rows, model="A"):
    Path(path).write_text(json.dumps({"model": model, "strategy": "sliding", "window": 20, "thinking": "off",
                                      "suffix_budget": BUDGET, "per_item": rows}))
    return Path(path)


def script(name, *args, cwd=ROOT):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / name), *map(str, args)],
                          capture_output=True, text=True, cwd=cwd)


# ---------------------------------------------------------------------------
# The exclusion rule

class TestExclusionRule(Base):
    def test_what_counts_as_an_infrastructure_failure(self):
        f = grader.is_infrastructure_failure
        legacy_row = {"billing": {"n_calls": 0}}                   # an earlier output: nothing billed
        self.assertTrue(f({"predicted": None, "raw_excerpt": "ERR:litellm.Timeout: Connection timed out"}, legacy_row))
        self.assertTrue(f({"predicted": None, "raw_excerpt": 'ERR:litellm.BadRequestError: {"message":"Insufficient Balance"}'},
                          legacy_row))
        self.assertTrue(f({"predicted": None, "raw_excerpt": "", "infra_error": "401 unauthorized"}))
        self.assertTrue(f({"raw": "ERR:litellm.ServiceUnavailableError: x"}, legacy_row))   # phase B / C record
        self.assertFalse(f({"predicted": None, "raw_excerpt": ""}, legacy_row))           # billed empty response
        self.assertFalse(f({"predicted": None, "raw_excerpt": "I can't help with that."}, legacy_row))  # refusal
        self.assertFalse(f({"predicted": None, "raw_excerpt": '{"balance": 12'}, legacy_row))  # truncation
        self.assertFalse(f({"predicted": None, "raw_excerpt": "ERR: model text"}, legacy_row))  # model text
        self.assertFalse(f({"predicted": {"error": {"code": "RATE_LIMITED"}}, "raw_excerpt": '{"error": 429}'}, legacy_row))
        self.assertFalse(f(None))

    def test_scoring_view(self):
        row = rows_for(CORE[:1])[0]
        self.assertIs(grader.scoring_view(row), row)
        fail_steps([row], [(0, 3), (0, 7)])
        row["phase_b"] = {"raw": "ERR:litellm.Timeout: timed out", "applicable": True}
        row["billing"] = {"n_calls": 38}                              # the three failed calls were not billed
        v = grader.scoring_view(row)
        self.assertEqual((v["suffix_len"], v["n_value_match"], v["n_well_formed"]), (38, 38, 38))
        self.assertEqual([s["suffix_idx"] for s in v["phase_a"]], [j for j in range(40) if j not in (3, 7)])
        self.assertIsNone(v["phase_b"])
        self.assertEqual(v["n_infra_excluded"], {"phase_a": 2, "phase_b": 1, "phase_c": 0})

    def test_every_consumer_excludes_failed_calls_and_reports_them(self):
        core = legacy(self.tmp / "core.json", fail_steps(rows_for(CORE), [(0, 0), (0, 1), (5, 2), (9, 39)]))
        ne = legacy(self.tmp / "ne.json", rows_for(NE))
        res = quiet(sp.score_pair, "A", core, ne, CORE, NE, BUDGET, 0, 17, False)
        self.assertEqual(res["infrastructure_failures_excluded"]["core_run_steps"], 4)
        self.assertEqual(res["effect_step_vm"]["value"], 100.0)          # the remaining predictions are exact
        self.assertEqual(res["diagnostics"]["core_all_call_vm"], 100.0)
        a = ar.audit_file(core, CORE, "schedule")
        self.assertEqual(a["infrastructure_failures_excluded"]["phase_a_steps"], 4)
        self.assertEqual((a["scored_calls"], a["all_call_vm"]), (10200 - 4, 100.0))
        self.assertEqual(a["outcomes_pct_of_calls"]["parse_fail"], 0.0)
        rows = lb.aggregate([core])
        self.assertEqual(sum(r["suffix_total"] for r in rows), 10200 - 4)
        self.assertEqual(sum(r["value_match"] for r in rows), 10200 - 4)
        self.assertEqual(sum(r["infra_excluded_calls"] for r in rows), 4)
        out = self.tmp / "v2"
        res_dir = self.tmp / "res"
        res_dir.mkdir()
        shutil.copy(core, res_dir / "core.json")
        r = script("apply_v2_filter.py", "--catalog", CORE_PATH, "--results-dir", res_dir, "--out-dir", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        summ = json.loads((out / "v2_filtered_leaderboard.json").read_text())["core"]["summary_all_items"]
        self.assertEqual((summ["infra_excluded_calls"], summ["vm"]), (4, 1.0))

    def test_empty_responses_and_refusals_are_still_scored(self):
        rows = rows_for(CORE)
        for j, raw in ((0, ""), (1, "I cannot predict this.")):
            rows[0]["phase_a"][j].update({"predicted": None, "value_match": False, "status_match": False,
                                          "well_formed": False, "raw_excerpt": raw})
        a = ar.audit_file(legacy(self.tmp / "core.json", rows), CORE, "schedule")
        self.assertEqual(a["infrastructure_failures_excluded"]["phase_a_steps"], 0)
        self.assertEqual(a["scored_calls"], 10200)
        self.assertLess(a["all_call_vm"], 100.0)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the released reference_results/")
    def test_released_counts(self):
        """The released audit reports, per run, exactly the provider errors found by the scan."""
        audit = {Path(x["file"]).name: x["infrastructure_failures_excluded"]
                 for x in json.loads((Path(REF) / "reference_audit.json").read_text())}
        self.assertEqual(len(audit), 26)
        expected = {"deepseek_v4_flash_tf1step_sliding_k20.json": (145, 5, 4),
                    "kimi_k25_tf1step_sliding_k20.json": (29, 0, 0),
                    "kimi_k25_no_suffix_feedback.json": (14, 1, 0),
                    "claude_sonnet46_tf1step_sliding_k20.json": (1, 0, 0),
                    "kimi_k25_adv24_tf1step_sliding_k20.json": (1, 0, 0),
                    "kimi_k25_original_run1_tf1step_sliding_k20.json": (38, 0, 0),
                    "kimi_k25_original_run2_tf1step_sliding_k20.json": (23, 0, 0),
                    "kimi_k25_renamed_run1_tf1step_sliding_k20.json": (42, 0, 0),
                    "kimi_k25_renamed_run2_tf1step_sliding_k20.json": (31, 0, 0)}
        for name, got in audit.items():
            with self.subTest(file=name):
                self.assertEqual((got["phase_a_steps"], got["phase_b_calls"], got["phase_c_calls"]),
                                 expected.get(name, (0, 0, 0)))
        # the 907 billed empty responses of the reasoning-on run are model outputs and stay scored
        run = json.loads((Path(REF) / "reasoning_raw" / "deepseek_v4_flash_thinking_no_suffix_feedback.json").read_text())
        empty = sum(1 for r in run["per_item"] for s in r["phase_a"] if not (s.get("raw_excerpt") or "").strip())
        self.assertEqual(empty, 907)
        self.assertEqual(audit["deepseek_v4_flash_thinking_no_suffix_feedback.json"]["phase_a_steps"], 0)


class TestBilledErrTextIsScored(Base):
    """Review round 11 (R13-B1): model output that starts with "ERR:" is scored."""

    def test_billed_refusal_through_eval_item(self):
        item = json.loads(CORE_PATH.read_text().splitlines()[0])
        answers = iter(["ERR: I cannot provide that prediction."]
                       + [json.dumps(s["response"]) for s in item["trace_counterfactual"][1:40]])

        def completion(*_a, **_k):   # every call returns billed model text
            usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            msg = types.SimpleNamespace(content=next(answers))
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)
        lit = types.SimpleNamespace(completion=completion, completion_cost=lambda **_k: 0.0)
        with mock.patch.dict(os.environ, {"CTE_LLM_RETRIES": "0"}):
            row = E.eval_item(dict(item), "test/model", "sliding", 20, lit, 40, False, False, "off", 4000, "high",
                              None, prompt_version="v1")
        self.assertEqual(row["billing"]["n_calls"], 40)
        self.assertEqual(row["n_infra_failures"], {"phase_a": 0, "phase_b": 0, "phase_c": 0})
        self.assertFalse(any(grader.is_infrastructure_failure(s, row) for s in row["phase_a"]))
        view = grader.scoring_view(row)
        self.assertIs(view, row)
        self.assertEqual((view["n_value_match"], view["suffix_len"]), (39, 40))          # 97.5%, not 100%
        out = legacy(self.tmp / "one.json", [row])
        a = ar.audit_file(out, [item], "schedule")
        self.assertEqual((a["scored_calls"], a["all_call_vm"]), (40, 97.5))
        self.assertEqual(a["infrastructure_failures_excluded"]["phase_a_steps"], 0)

    def test_legacy_rows_need_the_exception_record_and_no_billed_output(self):
        def row(raw, n_calls):
            steps = [{"suffix_idx": 0, "predicted": None, "raw_excerpt": raw},
                     {"suffix_idx": 1, "predicted": {"ok": True}, "raw_excerpt": "{}"}]
            return {"phase_a": steps, "billing": {"n_calls": n_calls}}
        f = grader.is_infrastructure_failure
        r = row("ERR:litellm.ServiceUnavailableError: BedrockException - unavailable", 1)
        self.assertTrue(f(r["phase_a"][0], r))                                   # exception record, unbilled
        r = row("ERR:litellm.ServiceUnavailableError: BedrockException - unavailable", 2)
        self.assertFalse(f(r["phase_a"][0], r))                                  # billed: model output
        r = row("ERR: I cannot provide that prediction.", 1)
        self.assertFalse(f(r["phase_a"][0], r))                                  # not the exception record
        r = row("ERR:litellm.Timeout: Connection timed out after None seconds.", 1)
        r["n_infra_failures"] = {"phase_a": 0, "phase_b": 0, "phase_c": 0}
        self.assertFalse(f(r["phase_a"][0], r))                                  # explicit records decide
        r = row("ERR:litellm.Timeout: Connection timed out after None seconds.", 1)
        del r["billing"]
        self.assertTrue(f(r["phase_a"][0], r))                                   # no billing recorded
        self.assertFalse(f(r["phase_a"][0]))                                     # the fallback needs the row


# ---------------------------------------------------------------------------
# F1, F7: score_paired

class TestPairedInputs(Base):
    def _pair(self, core_model, ne_model, label="A"):
        core = legacy(self.tmp / "core.json", rows_for(CORE), core_model)
        ne = legacy(self.tmp / "ne.json", rows_for(NE), ne_model)
        return script("score_paired.py", "--pair", label, core, ne)

    def test_runs_of_different_models_are_not_a_pair(self):
        r = self._pair("deepseek/deepseek-chat", "bedrock/moonshotai.kimi-k2.5", "DeepSeek V4-Flash")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("a pair must be one model", r.stderr)
        self.assertEqual(self._pair("DS-V4-Lite/sliding", "deepseek/deepseek-chat", "DeepSeek V4-Flash").returncode, 0)
        r = self._pair("deepseek/deepseek-chat", "deepseek/deepseek-chat", "Kimi K2.5")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("names another model", r.stderr)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the released reference_results/")
    def test_released_deepseek_core_with_kimi_no_effect(self):
        ref = Path(REF)
        r = script("score_paired.py", "--pair", "DeepSeek V4-Flash", ref / "raw/deepseek_v4_flash_tf1step_sliding_k20.json",
                   ref / "no_effect_v1_raw/kimi_k25_no_effect_v1_tf1step_sliding_k20.json")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("a pair must be one model", r.stderr)

    def test_duplicate_pair_labels_are_rejected(self):
        core = legacy(self.tmp / "core.json", rows_for(CORE))
        ne = legacy(self.tmp / "ne.json", rows_for(NE))
        out = self.tmp / "out.json"
        r = script("score_paired.py", "--pair", "A", core, ne, "--pair", "A", core, ne, "--json", out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("duplicate pair label", r.stderr)
        self.assertFalse(out.exists())


# ---------------------------------------------------------------------------
# F2: build_core_v1 never replaces outputs unless the build succeeds

class TestCoreBuilderOutputs(Base):
    def setUp(self):
        super().setUp()
        self.cf = self.tmp / "counterfactuals"
        shutil.copytree(ROOT / "counterfactuals", self.cf)
        self.frozen = (self.cf / "core_v1.jsonl").read_bytes()

    def _pool(self):
        for d in {r["domain"] for r in CORE}:
            (self.cf / f"{d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in CORE if r["domain"] == d))

    def _results(self, kimi_model="Kimi-K2.5/sliding"):
        res = self.tmp / "results"
        res.mkdir(exist_ok=True)
        for name, model in (("ds_lite_sliding_labeled.json", "DS-V4-Lite/sliding"),
                            ("kimi_k25_sliding_labeled.json", kimi_model),
                            ("sonnet46_sliding.json", "bedrock/global.anthropic.claude-sonnet-4-6")):
            legacy(res / name, rows_for(CORE), model)
        return res

    def test_default_command_on_the_release_tree_touches_nothing(self):
        r = script("build_core_v1.py", cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("is not the candidate pool", r.stderr)
        self.assertEqual((self.cf / "core_v1.jsonl").read_bytes(), self.frozen)
        self.assertFalse((self.tmp / "build").exists())

    def test_a_failing_build_leaves_the_catalog_and_no_temporary_files(self):
        self._pool()
        r = script("build_core_v1.py", "--out-jsonl", "counterfactuals/core_v1.jsonl", cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)                          # reference runs missing
        self.assertIn("reference runs missing", r.stderr)
        self._results(kimi_model="A")                                 # a run that is not the reference run
        r = script("build_core_v1.py", "--results-dir", "results", "--out-jsonl", "counterfactuals/core_v1.jsonl",
                   "--audit-json", "audit.json", cwd=self.tmp)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual((self.cf / "core_v1.jsonl").read_bytes(), self.frozen)
        self.assertFalse((self.tmp / "audit.json").exists())
        self.assertEqual(sorted(p.name for p in self.cf.glob(".*")), [])

    def test_a_successful_build_replaces_the_output(self):
        self._pool()
        self._results()
        target = self.tmp / "out" / "core.jsonl"
        target.parent.mkdir()
        target.write_text("old\n")
        r = script("build_core_v1.py", "--results-dir", "results", "--out-jsonl", target, "--audit-json",
                   self.tmp / "out" / "audit.json", cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(target.read_bytes(), self.frozen)
        self.assertEqual(sorted(p.name for p in target.parent.iterdir()), ["audit.json", "core.jsonl"])


# ---------------------------------------------------------------------------
# F3: build_no_effect_v1 needs the Core-v1 catalog

class TestNoEffectBuilderInput(Base):
    def test_other_catalog_is_rejected_before_building(self):
        out, rep = self.tmp / "ne.jsonl", self.tmp / "report.md"
        r = script("build_no_effect_v1.py", "--core-jsonl", NE_PATH, "--out-jsonl", out, "--report", rep)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("is not the Core-v1 catalog", r.stderr)
        self.assertFalse(out.exists() or rep.exists())


# ---------------------------------------------------------------------------
# F4, F5: the evaluator (offline stand-ins for litellm)

def _litellm(fail=False):
    def completion(*_a, **_k):
        if fail:
            raise ValueError("offline stand-in: provider returned an error")
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))],
                                     usage=usage)
    return types.SimpleNamespace(completion=completion, completion_cost=lambda **_k: 0.0)


class TestEvaluator(Base):
    def setUp(self):
        super().setUp()
        self.cf = self.tmp / "two.jsonl"
        self.cf.write_text("\n".join(CORE_PATH.read_text().splitlines()[:2]) + "\n")
        self.out = self.tmp / "run.json"

    def _main(self, n_items, model="openrouter/test/model", fail=False, env=(), *extra):
        argv = ["eval_llm_counterfactual.py", "--cf-jsonl", str(self.cf), "--model", model, "--strategy", "sliding",
                "--window", "20", "--max-items", str(n_items), "--suffix-budget", "40", "--no-explain", "--no-state",
                "--prompt-version", "v1", "--out", str(self.out), *extra]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"litellm": _litellm(fail)}), \
                mock.patch.dict(os.environ, {"CTE_LLM_RETRIES": "0", **dict(env)}), contextlib.redirect_stdout(buf):
            E.main()
        return buf.getvalue()

    def test_routing_and_reasoning_settings_are_stamped_and_compared(self):
        a = {"OPENROUTER_PROVIDER": "parasail", "OPENROUTER_REASONING_MAX_TOKENS": "256"}
        b = {"OPENROUTER_PROVIDER": "together", "OPENROUTER_REASONING_MAX_TOKENS": "2048"}
        self._main(1, env=a)
        out = json.loads(self.out.read_text())
        rs = out["per_item"][0][E.ROW_CONFIG_KEY]["request_settings"]
        self.assertEqual((rs["OPENROUTER_PROVIDER"], rs["OPENROUTER_REASONING_MAX_TOKENS"], rs["temperature"]),
                         ("parasail", "256", "provider default"))
        with self.assertRaises(SystemExit) as cm:
            self._main(2, env=b)
        self.assertIn("request_settings", str(cm.exception))
        log = self._main(2, "openrouter/test/model", False, b, "--allow-checkpoint-mismatch")
        self.assertIn("NOTE: this output mixes rows", log)
        self.assertTrue(json.loads(self.out.read_text())["mixed_run_configurations"])

    def test_outputs_differing_only_in_routing_are_not_pooled(self):
        a = write_output(self.tmp / "a.json", CORE, config(CORE_PATH, request_settings={"OPENROUTER_PROVIDER": "parasail"}))
        b = write_output(self.tmp / "b.json", CORE, config(CORE_PATH, request_settings={"OPENROUTER_PROVIDER": "together"}),
                         correct=False)
        rows = lb.aggregate([a, b])
        self.assertEqual(len({r["group_id"] for r in rows}), 2)

    def test_provider_errors_are_recorded_and_retried(self):
        log = self._main(1, fail=True)
        out = json.loads(self.out.read_text())
        row = out["per_item"][0]
        self.assertTrue(all(s.get("infra_error") and s["predicted"] is None for s in row["phase_a"]))
        self.assertEqual(row["n_infra_failures"]["phase_a"], 40)
        self.assertEqual((out["infrastructure_failures"]["rows"], out["infrastructure_failures"]["phase_a"]), (1, 40))
        self.assertIn("excluded from scoring and retried", log)
        rows = lb.aggregate([self.out], catalogs=[self.cf])                 # not scored as model failures
        self.assertEqual((sum(r["suffix_total"] for r in rows), sum(r["infra_excluded_calls"] for r in rows)), (0, 40))
        log = self._main(1, fail=True)                                       # still failing: retried again
        self.assertIn("1 items pending, 1 of them retried", log)
        log = self._main(1)                                                  # provider back: the row is evaluated
        self.assertIn("1 items pending, 1 of them retried", log)
        out = json.loads(self.out.read_text())
        self.assertEqual(out["infrastructure_failures"]["rows"], 0)
        self.assertFalse(any(s.get("infra_error") for s in out["per_item"][0]["phase_a"]))
        self.assertIn("billing_of_retried_rows", out)
        self.assertIn("0 items pending", self._main(1))                       # a clean row is reused


# ---------------------------------------------------------------------------
# F6: state probe aggregates

class TestStateProbeAggregates(Base):
    def _run(self, n, fail=()):
        argv = ["run_state_probe_v2.py", "--cf-jsonl", str(CORE_PATH), "--model", "A", "--max-items", str(n),
                "--max-concurrency", "1", "--calls-dir", str(self.tmp / "calls"), "--out", str(self.tmp / "out.json")]
        calls = []

        def fake_call(model_id, system, prompt, *_a, **_k):
            calls.append(prompt)
            if len(calls) in fail:
                raise ValueError("provider error")
            return "{}", {"prompt_tokens": 100, "cached_tokens": 50}, 0.25
        with mock.patch.object(sys, "argv", argv), mock.patch.object(rsp, "call_llm", fake_call), \
                contextlib.redirect_stdout(io.StringIO()):
            rsp.main()
        return calls, json.loads((self.tmp / "out.json").read_text())

    def test_cached_and_new_records_are_aggregated_together(self):
        calls, out = self._run(2)
        self.assertEqual((len(calls), out["n_records"], out["total_cost_usd"], out["cache_hit_pct"]), (2, 2, 0.5, 50.0))
        calls, out = self._run(2)                                     # resume: nothing new
        self.assertEqual((len(calls), out["n_records"], out["total_cost_usd"], out["cache_hit_pct"]), (0, 2, 0.5, 50.0))
        self.assertEqual((out["invocation"]["n_calls"], out["invocation"]["cost_usd"]), (0, 0.0))
        calls, out = self._run(1)                                     # --max-items applies to the output
        self.assertEqual((out["n_records"], out["total_cost_usd"]), (1, 0.25))

    def test_failed_calls_are_retried(self):
        calls, out = self._run(2, fail={1})
        self.assertEqual(out["n_failed_calls"], 1)
        calls, out = self._run(2)
        self.assertEqual(len(calls), 1)
        self.assertEqual((out["n_records"], out["n_failed_calls"]), (2, 0))


# ---------------------------------------------------------------------------
# F8: apply_v2_filter keeps file identity

class TestFilterFileIdentity(Base):
    def test_basename_aliases_are_rejected(self):
        res = self.tmp / "res"
        res.mkdir()
        legacy(res / "model_labeled.json", rows_for(CORE))
        legacy(res / "model.json", rows_for(CORE, correct=False))
        out = self.tmp / "out"
        r = script("apply_v2_filter.py", "--catalog", CORE_PATH, "--results-dir", res, "--out-dir", out)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("both report as 'model'", r.stderr)
        self.assertFalse((out / "v2_filtered_leaderboard.json").exists())
        (res / "model.json").rename(res / "other.json")
        r = script("apply_v2_filter.py", "--catalog", CORE_PATH, "--results-dir", res, "--out-dir", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        agg = json.loads((out / "v2_filtered_leaderboard.json").read_text())
        self.assertEqual((agg["model"]["file"], agg["other"]["file"]), ("model_labeled.json", "other.json"))


# ---------------------------------------------------------------------------
# F9, F10: renamed variant

class TestRenamedVariantInputs(Base):
    def test_skipped_verification_is_reported_as_unverified(self):
        out, rep = self.tmp / "renamed.jsonl", self.tmp / "report.json"
        r = subprocess.run([sys.executable, "-I", "scripts/build_renamed_core.py", "--no-verify", "--out", str(out),
                            "--report", str(rep)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        report = json.loads(rep.read_text())
        self.assertIs(report["trace_verified"], False)
        self.assertIn("skipped", report["verification"])
        self.assertIn("UNVERIFIED", r.stdout)
        self.assertNotIn("TRACE-VERIFIED (output equality on evaluated traces): True", r.stdout)

    @unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the released reference_results/")
    def test_ambiguous_arm_files_are_rejected(self):
        root = self.tmp / "ds"
        (root / "reference_results" / "renamed_raw").mkdir(parents=True)
        (root / "counterfactuals").symlink_to(Path(REF).resolve().parent / "counterfactuals")
        raw = Path(REF) / "renamed_raw"
        for f in raw.glob("*.json"):
            (root / "reference_results" / "renamed_raw" / f.name).symlink_to(f)
        extra = root / "reference_results" / "renamed_raw" / "copy_original_run1_x.json"
        extra.symlink_to(raw / "kimi_k25_original_run2_tf1step_sliding_k20.json")
        r = script("score_renamed_variant.py", "--dataset-root", root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("arm original_run1: 2 files match", r.stderr)
        r = script("score_renamed_variant.py", "--dataset-root", root, "--arm",
                   f"original_run1={raw / 'kimi_k25_original_run1_tf1step_sliding_k20.json'}")
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
