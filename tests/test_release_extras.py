"""Release checks for the identifier-renamed variant, the reference policies,
and the reasoning off/on comparison.

Run from the repository root:

    python -m unittest discover -s tests -v

With CTE_REFERENCE_RESULTS=/path/to/hf/reference_results, the renamed-variant
study and the reasoning comparison are also recomputed from the released raw
outputs (reference_results/renamed_raw/, reference_results/reasoning_raw/).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import reference_policies as rp  # noqa: E402
import score_paired as sp  # noqa: E402


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


CORE = sp.load_jsonl(ROOT / "counterfactuals" / "core_v1.jsonl")
CORE_BY_ID = {r["item_id"]: r for r in CORE}
RENAMED = sp.load_jsonl(ROOT / "counterfactuals" / "core_v1_renamed.jsonl")


class TestRenamedVariant(unittest.TestCase):
    def test_rows_pair_with_the_source_edit_scenarios(self):
        source_edits = {r["item_id"] for r in CORE if r.get("patched_source")}
        self.assertEqual(len(RENAMED), 210)
        self.assertEqual(len(source_edits), 210)
        self.assertEqual({r["paired_item_id"] for r in RENAMED}, source_edits)
        for r in RENAMED:
            self.assertEqual(r["item_id"], r["paired_item_id"] + "__renamed")
            self.assertEqual(r["source_variant"], "identifier_randomized")

    def test_only_identifiers_change(self):
        for r in RENAMED:
            c = CORE_BY_ID[r["paired_item_id"]]
            m = r["identifier_map"]
            with self.subTest(item=r["item_id"]):
                self.assertEqual(len(set(m.values())), len(m), "identifier map must be one-to-one")
                self.assertEqual(r["prefix_len"], c["prefix_len"])
                self.assertEqual(r["domain"], c["domain"])
                self.assertEqual(r["intervention"]["line_number"], c["intervention"]["line_number"])
                self.assertEqual(r["patched_source"].count("\n"), c["patched_source"].count("\n"))
                for key in ("trace_prefix", "trace_original", "trace_counterfactual"):
                    self.assertEqual(len(r[key]), len(c[key]))
                    for a, b in zip(r[key], c[key]):
                        self.assertEqual(a["op"], m.get(b["op"], b["op"]))
                        self.assertEqual(canon(a["args"]), canon(b["args"]))
                        self.assertEqual(canon(a["response"]), canon(b["response"]))

    def test_rebuild_is_byte_identical_and_trace_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, rep = Path(tmp) / "renamed.jsonl", Path(tmp) / "report.json"
            subprocess.run([sys.executable, "-I", "scripts/build_renamed_core.py", "--out", str(out),
                            "--report", str(rep)], cwd=ROOT, check=True, capture_output=True)
            self.assertEqual(out.read_bytes(), (ROOT / "counterfactuals" / "core_v1_renamed.jsonl").read_bytes())
            report = json.loads(rep.read_text())
            self.assertEqual(report["verified_steps"], 13940)
            self.assertEqual(report["mismatches"], 0)
            self.assertEqual(report["items_written"], 210)


class TestReferencePolicies(unittest.TestCase):
    EXPECTED = {  # (protocol, policy): (effect-step VM hits, all-call VM hits, SM hits); 2,476 effect steps, 10,200 calls
        ("tf1step_sliding_k20", "replay_without_intervention"): (0, 7724, 9346),
        ("tf1step_sliding_k20", "copy_last_same_op"): (353, 1178, 8541),
        ("tf1step_sliding_k20", "copy_last_same_call"): (478, 1890, 8179),
        ("no_suffix_feedback", "copy_last_same_op"): (44, 359, 7946),
        ("no_suffix_feedback", "copy_last_same_call"): (82, 944, 7957),
        ("free_rollout", "copy_last_same_op"): (35, 318, 7887),
        ("free_rollout", "copy_last_same_call"): (54, 404, 7915),
    }

    def test_reference_policy_scores(self):
        items = rp.load_items(ROOT / "counterfactuals" / "core_v1.jsonl")
        for (proto, name), (eff, vm, sm) in self.EXPECTED.items():
            res, _ = rp.score(items, rp.POLICIES[proto][name], n_boot=10, seed=17)
            with self.subTest(protocol=proto, policy=name):
                self.assertEqual((res["effect_step_vm"]["num"], res["effect_step_vm"]["den"]), (eff, 2476))
                self.assertEqual((res["all_call_vm"]["num"], res["all_call_vm"]["den"]), (vm, 10200))
                self.assertEqual(res["sm"]["num"], sm)

    def test_policies_see_exactly_the_prompt_history(self):
        """Each copy policy's prediction equals copying from the CALL lines of the
        evaluator's own v1 prompt for that step (same window, same history)."""
        import copy as _copy
        import eval_llm_counterfactual as E
        dec = json.JSONDecoder()

        def prompt_calls(prompt):
            calls = []
            for line in prompt.splitlines():
                if not line.startswith("CALL "):
                    continue
                rest = line.split(": ", 1)[1]
                op, rest = rest.split("(", 1)
                args, end = dec.raw_decode(rest)
                assert rest[end:end + 5] == ") -> "
                resp, _ = dec.raw_decode(rest[end + 5:])
                calls.append((op, canon(args), resp))
            return calls

        strategy_of = {"tf1step_sliding_k20": "sliding", "no_suffix_feedback": "prefix_only",
                       "free_rollout": "free_rollout"}
        items = rp.load_items(ROOT / "counterfactuals" / "core_v1.jsonl")
        n = 0
        for proto, strategy in strategy_of.items():
            for name in ("copy_last_same_op", "copy_last_same_call"):
                keyed = name == "copy_last_same_call"
                for row, it in zip(CORE, items):
                    item = E.normalize_line_number_convention(_copy.deepcopy(row))
                    item["source_code"] = ""
                    preds = rp.POLICIES[proto][name](it)
                    hist = []
                    for j, s in enumerate(item["trace_counterfactual"][:rp.BUDGET]):
                        prompt = E.make_counterfactual_prompt(item, hist, s["op"], s["args"], strategy, rp.WINDOW)
                        key = (s["op"], canon(s["args"])) if keyed else s["op"]
                        want = {}
                        for op, a, resp in reversed(prompt_calls(prompt)):
                            if ((op, a) if keyed else op) == key:
                                want = resp
                                break
                        self.assertEqual(canon(preds[j]), canon(want), (proto, name, row["item_id"], j))
                        n += 1
                        resp = s["response"] if strategy == "sliding" else preds[j]
                        hist.append({"op": s["op"], "args": s["args"], "response": resp})
        self.assertEqual(n, 3 * 2 * 255 * rp.BUDGET)


REF = os.environ.get("CTE_REFERENCE_RESULTS")


@unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the HF reference_results/ directory")
class TestReleasedStudies(unittest.TestCase):
    def _run(self, script: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out.json"
            subprocess.run([sys.executable, f"scripts/{script}", "--dataset-root", str(Path(REF).resolve().parent),
                            "--json", str(out)], cwd=ROOT, check=True, capture_output=True)
            return json.loads(out.read_text())

    def test_renamed_variant_study(self):
        d = self._run("score_renamed_variant.py")
        self.assertEqual((d["scenarios"], d["scenarios_with_effect_steps"], d["effect_steps"]), (210, 202, 2040))
        e, a = d["effect_step_vm"], d["agreement"]
        got = {k: round(v["value"], 2) for k, v in {**e, **{f"agree_{k}": v for k, v in a.items()}}.items()}
        # calls that returned a provider error (Bedrock service unavailable) are excluded
        self.assertEqual(got, {"original_run1": 64.5, "original_run2": 63.32, "renamed_run1": 60.73,
                               "renamed_run2": 61.75, "renamed_minus_original": -2.67,
                               "agree_original_repeat": 65.43, "agree_renamed_repeat": 60.61, "agree_cross": 61.01,
                               "agree_cross_minus_renamed_repeat": 0.4, "agree_renamed_minus_original_repeat": -4.82})
        self.assertEqual([round(x, 2) for x in a["original_repeat"]["ci95"]], [62.52, 68.21])
        self.assertEqual([round(x, 2) for x in a["cross_minus_renamed_repeat"]["ci95"]], [-1.3, 1.97])
        self.assertEqual({k: v for k, v in d["infrastructure_failures_excluded"].items() if k != "rule"},
                         {"original_run1": 38, "original_run2": 23, "renamed_run1": 42, "renamed_run2": 31})

    def test_reasoning_pairs(self):
        d = self._run("score_reasoning_pairs.py")
        # the off run with answers revealed has 145 calls that returned a provider error (excluded)
        expected = {"tf1step_sliding_k20": (233, 2276, 2296, 62.26, 81.05),
                    "no_suffix_feedback": (255, 2476, 2476, 24.60, 34.49)}
        for proto, (n, eff_off, eff_on, off, on) in expected.items():
            p = d["protocols"][proto]
            with self.subTest(protocol=proto):
                self.assertEqual(p["scenarios"], n)
                self.assertEqual(p["off"]["effect_steps"], eff_off)
                self.assertEqual(p["on"]["effect_steps"], eff_on)
                self.assertAlmostEqual(p["off"]["effect_step_vm"], off, places=2)
                self.assertAlmostEqual(p["on"]["effect_step_vm"], on, places=2)
                self.assertEqual(p["on"]["incomplete_rows"], 0)


if __name__ == "__main__":
    unittest.main()
