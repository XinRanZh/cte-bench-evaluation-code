"""Release checks for the no-effect-v1 companion split and the paired scorer.

Run from the repository root:

    python -m unittest discover -s tests -v        # or: python -m pytest tests -v

Optional checks:

    CTE_REFERENCE_RESULTS=/path/to/hf/reference_results  python -m unittest discover -s tests -v
        reproduces the published DeepSeek V4-Flash / Kimi K2.5 paired numbers from
        the raw reference outputs released on Hugging Face
        (reference_results/raw/ and reference_results/no_effect_v1_raw/).

    CTE_SLOW_TESTS=1  python -m unittest discover -s tests -v
        rebuilds counterfactuals/no_effect_v1.jsonl with scripts/build_no_effect_v1.py
        (about 15 s, CPU only) and checks that the output is byte-identical.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import core_v1  # noqa: E402
import score_paired as sp  # noqa: E402

BUDGET = 40


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True)


CORE = core_v1.load_core_v1()
NE = core_v1.load_no_effect_v1()
CORE_BY_ID = {r["item_id"]: r for r in CORE}


class TestSplit(unittest.TestCase):
    def test_row_counts(self):
        self.assertEqual(len(CORE), 255)
        self.assertEqual(len(NE), 255)
        ids = [r["item_id"] for r in NE]
        self.assertEqual(len(set(ids)), 255)
        self.assertFalse(set(ids) & set(CORE_BY_ID), "companion ids must not reuse Core-v1 ids")

    def test_pairing_is_one_to_one(self):
        paired = [r["paired_item_id"] for r in NE]
        self.assertEqual(len(set(paired)), 255)
        self.assertEqual(set(paired), set(CORE_BY_ID))
        self.assertEqual(core_v1.pair_index(NE).keys(), CORE_BY_ID.keys())
        for r in NE:
            partner = CORE_BY_ID[r["paired_item_id"]]
            self.assertEqual(r["domain"], partner["domain"])
            self.assertEqual(r["paired_core_kind"], partner["intervention"]["kind"])

    def test_prefix_and_schedule_identical_to_partner(self):
        for r in NE:
            partner = CORE_BY_ID[r["paired_item_id"]]
            self.assertEqual(r["prefix_len"], partner["prefix_len"])
            self.assertEqual(len(r["trace_prefix"]), r["prefix_len"])
            self.assertEqual(canon(r["trace_prefix"]), canon(partner["trace_prefix"]))
            sched = [(s["op"], canon(s["args"])) for s in r["trace_counterfactual"][:BUDGET]]
            sched_p = [(s["op"], canon(s["args"])) for s in partner["trace_counterfactual"][:BUDGET]]
            self.assertEqual(sched, sched_p)
            self.assertEqual(len(sched), BUDGET)

    def test_all_scored_positions_identical_across_the_two_executions(self):
        positions = differ = 0
        for r in NE:
            for a, b in zip(r["trace_original"][:BUDGET], r["trace_counterfactual"][:BUDGET]):
                positions += 1
                differ += canon(a["response"]) != canon(b["response"])
        self.assertEqual(positions, 10_200)
        self.assertEqual(differ, 0)

    def test_full_stored_suffix_is_response_identical(self):
        for r in NE:
            self.assertEqual(len(r["trace_original"]), len(r["trace_counterfactual"]))
            self.assertEqual([canon(s["response"]) for s in r["trace_original"]],
                             [canon(s["response"]) for s in r["trace_counterfactual"]],
                             r["item_id"])

    def test_cells_match_core_by_pair(self):
        partner_cells = Counter(core_v1.catalog_cell(CORE_BY_ID[r["paired_item_id"]]) for r in NE)
        core_cells = Counter(core_v1.catalog_cell(r) for r in CORE)
        self.assertEqual(partner_cells, core_cells)
        self.assertEqual(set(core_cells), core_v1.CORE_V1_CELL_SET)
        self.assertEqual(sorted(core_cells.values()), [15] + [30] * 8)

    def test_emitted_cells_and_labels(self):
        emitted = {core_v1.catalog_cell(r) for r in NE}
        self.assertEqual(emitted, core_v1.NO_EFFECT_V1_CELL_SET)
        self.assertTrue(all(core_v1.is_no_effect_v1_catalog_item(r) for r in NE))
        self.assertEqual(Counter(r["benign_tier"] for r in NE), {"same_kind": 162, "near_kind": 93})
        self.assertEqual(Counter(r["contingency_class"] for r in NE), {"global_no_op": 141, "contingent": 114})
        self.assertEqual(sum(bool(r["latent_divergence"]) for r in NE), 10)
        self.assertEqual(Counter(r["intervention"]["kind"] for r in NE),
                         {"threshold_change": 144, "branch_inversion": 96, "state_injection": 15})

    def test_core_effect_steps_frozen(self):
        eff = scen = 0
        for r in CORE:
            n = sum(canon(a["response"]) != canon(b["response"])
                    for a, b in zip(r["trace_original"][:BUDGET], r["trace_counterfactual"][:BUDGET]))
            eff += n
            scen += n > 0
        self.assertEqual((eff, scen), (2476, 247))

    def test_coverage_audit(self):
        audit = sp.coverage_audit(CORE, NE)
        self.assertEqual(audit["tvd"]["domain"], 0.0)
        self.assertEqual(audit["tvd"]["prefix_group"], 0.0)
        self.assertEqual(audit["tvd"]["trajectory"], 0.0)
        self.assertAlmostEqual(audit["tvd"]["intervention_kind"], 0.3294, places=4)
        pairing = sp.validate_pairing(CORE, NE, BUDGET)
        self.assertEqual(pairing["problems"], [])
        self.assertTrue(pairing["pairs_1to1"])


class TestChecksums(unittest.TestCase):
    def test_sha256sums_match(self):
        sums = ROOT / "SHA256SUMS.txt"
        bad = []
        for line in sums.read_text().splitlines():
            digest, rel = line.split(None, 1)
            path = ROOT / rel.strip()
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                bad.append(rel)
        self.assertEqual(bad, [])


def _write_run(path: Path, rows: list[dict], predict) -> None:
    per_item = []
    for r in rows:
        steps = []
        for j, s in enumerate(r["trace_counterfactual"][:BUDGET]):
            pred = predict(r, j)
            steps.append({"suffix_idx": j, "op": s["op"], "args": s["args"],
                          "predicted": pred, "ground_truth": s["response"],
                          "value_match": pred == s["response"],
                          "status_match": True, "well_formed": isinstance(pred, dict)})
        per_item.append({"item_id": r["item_id"], "domain": r["domain"],
                         "intervention_kind": r["intervention"]["kind"], "phase_a": steps})
    path.write_text(json.dumps({"model": "synthetic", "strategy": "sliding", "window": 20,
                                "thinking": "off", "suffix_budget": BUDGET,
                                "n_items": len(per_item), "per_item": per_item}))


class TestScorerSynthetic(unittest.TestCase):
    """Degenerate predictors whose paired scores are known in closed form."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _score(self, core_pred, ne_pred, ne_rows=None, allow_partial=False):
        c, n = self.dir / "core.json", self.dir / "ne.json"
        _write_run(c, CORE, core_pred)
        _write_run(n, ne_rows if ne_rows is not None else NE, ne_pred)
        return sp.score_pair("synthetic", c, n, CORE, NE, BUDGET, 200, 17, allow_partial)

    def test_null_world_predictor(self):
        null = lambda r, j: r["trace_original"][j]["response"]  # noqa: E731
        res = self._score(null, null)
        self.assertTrue(res["complete"])
        self.assertEqual(res["effect_step_vm"]["value"], 0.0)
        self.assertEqual(res["effect_step_vm"]["den"], 2476)
        self.assertEqual(res["no_effect_vm"]["value"], 100.0)
        self.assertEqual(res["no_effect_over_prediction_rate"]["value"], 0.0)

    def test_oracle_predictor(self):
        oracle = lambda r, j: r["trace_counterfactual"][j]["response"]  # noqa: E731
        res = self._score(oracle, oracle)
        self.assertEqual(res["effect_step_vm"]["value"], 100.0)
        self.assertEqual(res["no_effect_vm"]["value"], 100.0)

    def test_always_change_predictor(self):
        oracle = lambda r, j: r["trace_counterfactual"][j]["response"]  # noqa: E731
        changed = lambda r, j: {"error": {"code": "INVENTED"}}  # noqa: E731
        res = self._score(oracle, changed)
        self.assertEqual(res["no_effect_vm"]["value"], 0.0)
        self.assertEqual(res["no_effect_over_prediction_rate"]["value"], 100.0)

    # Malformed or flag-only runs must never be scored.

    def _tampered(self, mutate, side="ne"):
        oracle = lambda r, j: r["trace_counterfactual"][j]["response"]  # noqa: E731
        c, n = self.dir / "core.json", self.dir / "ne.json"
        _write_run(c, CORE, oracle)
        _write_run(n, NE, oracle)
        target = n if side == "ne" else c
        run = json.loads(target.read_text())
        mutate(run)
        target.write_text(json.dumps(run))
        return lambda: sp.score_pair("tampered", c, n, CORE, NE, BUDGET, 0, 17, False)

    def test_duplicate_suffix_indices_rejected(self):
        def dup(run):  # 40 copies of one correct step per item
            for item in run["per_item"]:
                item["phase_a"] = [dict(item["phase_a"][0]) for _ in range(BUDGET)]
        with self.assertRaises(sp.InvalidRunError):
            self._tampered(dup)()

    def test_flag_only_steps_rejected(self):
        def strip(run):  # no predictions, all stored flags true (finding B2)
            for item in run["per_item"]:
                for st in item["phase_a"]:
                    st.pop("predicted")
                    st.update(value_match=True, well_formed=True, status_match=True)
        for side in ("ne", "core"):
            with self.subTest(side=side), self.assertRaises(sp.InvalidRunError):
                self._tampered(strip, side)()

    def test_bad_indices_rejected(self):
        for bad in (-1, 10_000, "3", None, True):
            def mutate(run, bad=bad):
                run["per_item"][0]["phase_a"][0]["suffix_idx"] = bad
            with self.subTest(bad=bad), self.assertRaises(sp.InvalidRunError):
                self._tampered(mutate)()

    def test_call_mismatch_rejected(self):
        def mutate(run):
            run["per_item"][0]["phase_a"][5]["args"] = {"tampered": 1}
        with self.assertRaises(sp.InvalidRunError):
            self._tampered(mutate)()

    def test_missing_step_is_incomplete(self):
        def mutate(run):
            run["per_item"][0]["phase_a"] = run["per_item"][0]["phase_a"][:-1]
        with self.assertRaises(SystemExit):
            self._tampered(mutate)()

    def test_partial_run_is_refused_unless_allowed(self):
        oracle = lambda r, j: r["trace_counterfactual"][j]["response"]  # noqa: E731
        with self.assertRaises(SystemExit):
            self._score(oracle, oracle, ne_rows=NE[:10])
        res = self._score(oracle, oracle, ne_rows=NE[:10], allow_partial=True)
        self.assertFalse(res["complete"])
        self.assertEqual(res["pairs_scored"], "10/255")


class TestPromptV1Regression(unittest.TestCase):
    """Prompt-v1 regression test: the prompts built for Core-v1 and no-effect-v1
    are pinned by digest, so results produced with prompt v1 stay reproducible.
    Prompt changes ship under a new protocol version."""

    V1_DIGEST = "aaad189f5b24282309614e3f3a40daf6f12ef91e9faba9396d5f7c33f711018f"

    def test_v1_prompt_digest(self):
        import copy
        import eval_llm_counterfactual as E
        h = hashlib.sha256()
        n = 0
        for rows in (CORE, NE):
            for row in rows:
                item = copy.deepcopy(row)
                E._attach_original_source(item)
                for strategy in ("sliding", "prefix_only", "free_rollout"):
                    hist = []
                    for s in item["trace_counterfactual"][:BUDGET]:
                        h.update(E.make_counterfactual_prompt(item, hist, s["op"], s["args"],
                                                              strategy, 20).encode())
                        n += 1
                        hist.append({"op": s["op"], "args": s["args"], "response": s["response"]})
                for fn in (E.make_explain_prompt, E.make_state_prompt):
                    try:
                        p = fn(item)
                    except Exception as e:  # noqa: BLE001 - digest records the failure mode
                        p = f"EXC:{type(e).__name__}"
                    h.update(str(p).encode())
                    n += 1
        self.assertEqual(n, 62220)
        self.assertEqual(h.hexdigest(), self.V1_DIGEST)

    def test_source_edit_intervention_block(self):
        import eval_llm_counterfactual as E
        block = E._fmt_intervention({"kind": "threshold_change", "line_number": 9, "col_offset": 0,
                                     "original_span": "0", "modified_span": "1", "description": "d",
                                     "step_k": 5})
        self.assertTrue(block.startswith("INTERVENTION (applied at step 9):"))


REF = os.environ.get("CTE_REFERENCE_RESULTS")


@unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the HF reference_results/ directory")
class TestReferenceReproduction(unittest.TestCase):
    """Published paired numbers (answers revealed / TF-1step sliding K=20).

    Calls that returned a provider error without model output are excluded
    (DeepSeek: 145 Core-v1 steps, insufficient balance; Kimi: 29 steps)."""

    EXPECTED = {
        "DeepSeek V4-Flash": dict(
            core="raw/deepseek_v4_flash_tf1step_sliding_k20.json",
            ne="no_effect_v1_raw/deepseek_v4_flash_no_effect_v1_tf1step_sliding_k20.json",
            eff=(1504, 2456), eff_scen=244, eff_ci=(58.1, 64.3), ne_vm=(8036, 10200), ne_ci=(77.5, 80.1),
            over=(2160, 10196), contingent=75.2, global_no_op=81.7, infra=145, null_retreat=(327, 952)),
        "Kimi K2.5": dict(
            core="raw/kimi_k25_tf1step_sliding_k20.json",
            ne="no_effect_v1_raw/kimi_k25_no_effect_v1_tf1step_sliding_k20.json",
            eff=(1522, 2467), eff_scen=247, eff_ci=(58.5, 64.7), ne_vm=(7647, 10200), ne_ci=(73.7, 76.3),
            over=(2507, 10154), contingent=72.8, global_no_op=76.7, infra=29, null_retreat=(268, 945)),
    }

    def test_paired_numbers(self):
        ref = Path(REF)
        for label, e in self.EXPECTED.items():
            with self.subTest(model=label):
                res = sp.score_pair(label, ref / e["core"], ref / e["ne"], CORE, NE,
                                    BUDGET, 2000, 17, False)
                self.assertTrue(res["complete"])
                eff, ne = res["effect_step_vm"], res["no_effect_vm"]
                self.assertEqual((eff["num"], eff["den"], eff["n_scenarios"]), (*e["eff"], e["eff_scen"]))
                self.assertEqual(res["infrastructure_failures_excluded"]["core_run_steps"], e["infra"])
                self.assertEqual(res["infrastructure_failures_excluded"]["no_effect_run_steps"], 0)
                nr = res["null_retreat"]
                self.assertEqual((nr["num"], nr["den"]), e["null_retreat"])
                self.assertEqual((ne["num"], ne["den"], ne["n_scenarios"]), (*e["ne_vm"], 255))
                self.assertEqual([round(x, 1) for x in eff["ci95"]], list(e["eff_ci"]))
                self.assertEqual([round(x, 1) for x in ne["ci95"]], list(e["ne_ci"]))
                over = res["no_effect_over_prediction_rate"]
                self.assertEqual((over["num"], over["den"]), e["over"])
                cc = res["no_effect_strata"]["contingency_class"]
                self.assertEqual(round(cc["contingent"]["no_effect_vm"]["value"], 1), e["contingent"])
                self.assertEqual(round(cc["global_no_op"]["no_effect_vm"]["value"], 1), e["global_no_op"])
                self.assertEqual(cc["contingent"]["no_effect_vm"]["n_scenarios"], 114)
                self.assertEqual(cc["global_no_op"]["no_effect_vm"]["n_scenarios"], 141)
                d = res["diagnostics"]
                self.assertEqual(d["core_rows_with_catalog_id_but_no_schedule_match"], 0)
                self.assertEqual(d["core_stored_flag_disagreements"], 0)
                self.assertEqual(d["no_effect_stored_flag_disagreements"], 0)
                self.assertEqual(d["no_effect_ground_truth_mismatches"], 0)


    def test_reference_audit(self):
        """Every released reference output covers its catalog, with no duplicate
        rows, no rows answering another scenario, and no stored-flag disagreements."""
        import audit_reference_results as ar
        ref = Path(REF)
        adv24 = core_v1.load_jsonl("counterfactuals/adv24_reasoning_probe.jsonl")
        groups = {"raw": (CORE, "schedule"), "no_effect_v1_raw": (NE, "item_id"),
                  "adv24_raw": (adv24, "schedule")}
        n_files = 0
        for sub, (cat, join) in groups.items():
            for path in sorted((ref / sub).glob("*.json")):
                n_files += 1
                r = ar.audit_file(path, cat, join)
                with self.subTest(file=f"{sub}/{path.name}"):
                    self.assertEqual(r["missing_scenarios"], [])
                    self.assertEqual(r["duplicate_rows_except_id"], [])
                    self.assertEqual(r["rows_answering_other_scenario"], [])
                    self.assertEqual(r["unmatched_rows"], [])
                    self.assertEqual(r["ambiguous"], [])
                    self.assertEqual(r["invalid_rows"], [])
                    self.assertEqual(r["incomplete_rows"], 0)
                    self.assertEqual(r["stored_flag_disagreements"], 0)
        self.assertEqual(n_files, 20)

    def test_tampered_reference_runs_rejected(self):
        """Two tampered variants of the real DeepSeek V4-Flash file are rejected."""
        ref = Path(REF)
        e = self.EXPECTED["DeepSeek V4-Flash"]
        base = json.loads((ref / e["ne"]).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("duplicates", "flag_only"):
                run = json.loads(json.dumps(base))
                for item in run["per_item"]:
                    if name == "duplicates":
                        good = next((s for s in item["phase_a"] if s["value_match"]), item["phase_a"][0])
                        item["phase_a"] = [dict(good) for _ in range(BUDGET)]
                    else:
                        for st in item["phase_a"]:
                            st.pop("predicted")
                            st.update(value_match=True, well_formed=True, status_match=True)
                path = Path(tmp) / f"{name}.json"
                path.write_text(json.dumps(run))
                with self.subTest(attack=name), self.assertRaises(sp.InvalidRunError):
                    sp.score_pair(name, ref / e["core"], path, CORE, NE, BUDGET, 0, 17, False)


@unittest.skipUnless(os.environ.get("CTE_SLOW_TESTS"), "set CTE_SLOW_TESTS=1 to rebuild the split")
class TestBuilderDeterminism(unittest.TestCase):
    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "no_effect_v1.jsonl"
            subprocess.run(
                [sys.executable, str(ROOT / "scripts" / "build_no_effect_v1.py"),
                 "--core-jsonl", str(ROOT / "counterfactuals" / "core_v1.jsonl"),
                 "--out-jsonl", str(out), "--report", str(Path(tmp) / "report.md"),
                 "--v0-jsonl", str(ROOT / "counterfactuals" / "no_effect_v0.jsonl"),
                 "--seed", "20260727"],
                check=True, cwd=tmp, capture_output=True)
            self.assertEqual(out.read_bytes(),
                             (ROOT / "counterfactuals" / "no_effect_v1.jsonl").read_bytes())


if __name__ == "__main__":
    unittest.main()
