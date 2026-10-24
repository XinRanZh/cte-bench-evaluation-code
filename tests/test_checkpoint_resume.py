"""Checkpoint resumption: the run configuration (prompt version, protocol,
window, model, catalog hash, ...) is recorded in the checkpoint header and on
every row, and a resume reuses rows only if all of them match.

The evaluator runs in-process with an offline stand-in for litellm whose
completion call returns the fixed model text "{}" at once, so no request
ever leaves the machine.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import eval_llm_counterfactual as E  # noqa: E402


def _offline_litellm():
    def completion(*_a, **_k):   # a model answer, not a provider error
        usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))],
                                     usage=usage)
    return types.SimpleNamespace(completion=completion, completion_cost=lambda **_k: 0.0)


class TestCheckpointResume(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cf = self.dir / "two.jsonl"
        self.cf.write_text("\n".join((ROOT / "counterfactuals" / "core_v1.jsonl").read_text().splitlines()[:2]) + "\n")
        self.items = [json.loads(l) for l in self.cf.read_text().splitlines()]
        self.out = self.dir / "run.json"
        self.ckpt = self.dir / "run.json.ckpt.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def _argv(self, prompt_version, n_items=1, *extra):
        return ["eval_llm_counterfactual.py", "--cf-jsonl", str(self.cf), "--model", "test/model",
                "--strategy", "sliding", "--window", "20", "--max-items", str(n_items), "--suffix-budget", "40",
                "--no-explain", "--no-state", "--prompt-version", prompt_version, "--out", str(self.out), *extra]

    def _main(self, argv) -> str:
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"litellm": _offline_litellm()}), \
                contextlib.redirect_stdout(buf):
            E.main()
        return buf.getvalue()

    def _config(self, prompt_version):
        ap = types.SimpleNamespace(cf_jsonl=str(self.cf), model="test/model", strategy="sliding", window=20,
                                   prompt_version=prompt_version, suffix_budget=40, thinking="off",
                                   reasoning_effort="high", max_tokens=4000, no_explain=True, no_state=True)
        return E.checkpoint_run_config(ap)

    def _row(self, stamp=None, k=0, identity=True):
        it = self.items[k]
        steps = [{"suffix_idx": j, "op": s["op"], "args": s["args"], "predicted": s["response"],
                  "ground_truth": s["response"], "value_match": True, "status_match": True, "well_formed": True}
                 for j, s in enumerate(it["trace_counterfactual"][:40])]
        row = {"item_id": it["item_id"], "domain": it["domain"], "intervention_kind": it["intervention"]["kind"],
               "prefix_len": it["prefix_len"], "suffix_len": 40, "n_well_formed": 40, "n_value_match": 40,
               "n_status_match": 40, "first_value_mismatch_at": None, "expected_delay": None,
               "billing": {"n_calls": 40}, "phase_a": steps, "phase_b": None, "phase_c": None}
        if stamp is not None:
            row[E.ROW_CONFIG_KEY] = self._config(stamp)
        if identity:
            import run_provenance as rp
            row[E.ROW_ID_KEY] = rp.catalog_row_id(it)
        return row

    def _write_ckpt(self, header=None, rows=()):
        lines = [] if header is None else [json.dumps({E.CHECKPOINT_CONFIG_KEY: self._config(header)})]
        self.ckpt.write_text("\n".join(lines + [json.dumps(r) for r in rows]) + "\n")

    def _ckpt_lines(self):
        return [json.loads(l) for l in self.ckpt.read_text().splitlines()]

    # -- direct mismatches -------------------------------------------------
    def test_resume_with_mismatched_prompt_version_fails(self):
        self._write_ckpt("v1", [self._row("v1")])
        with self.assertRaises(SystemExit) as cm:
            self._main(self._argv("v1.1"))
        self.assertIn("prompt_version: 'v1' vs this run 'v1.1'", str(cm.exception))
        self.assertFalse(self.out.exists())

    def test_resume_with_other_window_fails(self):
        self._write_ckpt("v1", [self._row("v1")])
        argv = self._argv("v1")
        argv[argv.index("--window") + 1] = "10"
        with self.assertRaises(SystemExit) as cm:
            self._main(argv)
        self.assertIn("window", str(cm.exception))

    def test_row_with_other_configuration_fails_even_if_header_matches(self):
        self._write_ckpt("v1.1", [self._row("v1")])
        with self.assertRaises(SystemExit) as cm:
            self._main(self._argv("v1.1"))
        self.assertIn("rows were not produced under this run's configuration", str(cm.exception))

    def test_unstamped_row_fails(self):
        self._write_ckpt("v1.1", [self._row(None)])
        with self.assertRaises(SystemExit) as cm:
            self._main(self._argv("v1.1"))
        self.assertIn("no recorded run configuration", str(cm.exception))

    def test_unverifiable_checkpoint_needs_override_and_stays_marked(self):
        self._write_ckpt(None, [self._row(None)])
        with self.assertRaises(SystemExit) as cm:
            self._main(self._argv("v1.1"))
        self.assertIn("records no run configuration", str(cm.exception))
        log = self._main(self._argv("v1.1", 1, "--allow-checkpoint-mismatch"))
        out = json.loads(self.out.read_text())
        self.assertTrue(out["mixed_run_configurations"])
        self.assertIn("NOTE: this output mixes rows", log)
        self.assertTrue(any(E.CHECKPOINT_MIXED_KEY in rec for rec in self._ckpt_lines()))
        with self.assertRaises(SystemExit) as cm:
            self._main(self._argv("v1.1"))
        self.assertIn("marked as mixing run configurations", str(cm.exception))

    # -- clean resumes -----------------------------------------------------
    def test_matching_resume_reuses_rows(self):
        self._write_ckpt("v1", [self._row("v1")])
        self._main(self._argv("v1"))
        out = json.loads(self.out.read_text())
        self.assertEqual(out["prompt_version"], "v1")
        self.assertFalse(out["mixed_run_configurations"])
        self.assertEqual(out["per_item"], [self._row("v1")])

    def test_clean_same_config_resume_extends_the_run(self):
        self._main(self._argv("v1.1", 1))
        self._main(self._argv("v1.1", 2))
        out = json.loads(self.out.read_text())
        self.assertFalse(out["mixed_run_configurations"])
        self.assertNotIn("row_run_configurations", out)
        self.assertEqual([r["item_id"] for r in out["per_item"]], [it["item_id"] for it in self.items])
        for r in out["per_item"]:
            self.assertEqual(r[E.ROW_CONFIG_KEY], self._config("v1.1"))
        lines = self._ckpt_lines()
        self.assertEqual(lines[0], {E.CHECKPOINT_CONFIG_KEY: self._config("v1.1")})
        self.assertEqual(len(lines), 3)  # header + one row per run
        self.assertFalse(any(E.CHECKPOINT_MIXED_KEY in rec for rec in lines))

    # -- the overridden-resume sequence ------------------------------------
    def test_overridden_resume_cannot_be_relabelled_later(self):
        # 1. v1.1, one item
        self._main(self._argv("v1.1", 1))
        # 2. v1, two items, with the override: item 1 is a v1.1 row, item 2 a new v1 row
        log2 = self._main(self._argv("v1", 2, "--allow-checkpoint-mismatch"))
        out2 = json.loads(self.out.read_text())
        self.assertTrue(out2["mixed_run_configurations"])
        self.assertIn("NOTE: this output mixes rows", log2)
        stamps = [r[E.ROW_CONFIG_KEY]["prompt_version"] for r in out2["per_item"]]
        self.assertEqual(stamps, ["v1.1", "v1"])
        self.assertEqual(sorted(c["run_config"]["prompt_version"] for c in out2["row_run_configurations"]),
                         ["v1", "v1.1"])
        self.assertTrue(any(E.CHECKPOINT_MIXED_KEY in rec for rec in self._ckpt_lines()))
        # 3. v1.1, two items, no override: must be refused, and must not overwrite the output
        with self.assertRaises(SystemExit) as cm:
            self._main(self._argv("v1.1", 2))
        msg = str(cm.exception)
        self.assertIn("rows were not produced under this run's configuration", msg)
        self.assertIn("marked as mixing run configurations", msg)
        self.assertEqual(json.loads(self.out.read_text()), out2)

    def test_new_checkpoint_starts_with_the_run_configuration(self):
        self._main(self._argv("v1.1"))
        first = self._ckpt_lines()[0]
        self.assertEqual(first, {E.CHECKPOINT_CONFIG_KEY: self._config("v1.1")})
        out = json.loads(self.out.read_text())
        self.assertEqual(out["prompt_version"], "v1.1")
        self.assertEqual(out["run_config"], self._config("v1.1"))


class TestCheckpointRowIdentity(unittest.TestCase):
    """Rows are keyed by catalog-row identity, so repeated item_ids (adv24: 24 rows,
    10 ids) restore each variant's own row, and a row is validated before reuse."""

    def setUp(self):
        import run_provenance as rp
        self.rp = rp
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.cf = ROOT / "counterfactuals" / "adv24_reasoning_probe.jsonl"
        self.items = [json.loads(l) for l in self.cf.read_text().splitlines()]
        self.out = self.dir / "adv.json"
        self.ckpt = self.dir / "adv.json.ckpt.jsonl"
        ns = types.SimpleNamespace(cf_jsonl=str(self.cf), model="test/model", strategy="sliding", window=20,
                                   prompt_version="v1", suffix_budget=40, thinking="off", reasoning_effort="high",
                                   max_tokens=4000, no_explain=True, no_state=True)
        self.config = E.checkpoint_run_config(ns)

    def tearDown(self):
        self.tmp.cleanup()

    def _oracle(self, it, uid_of=None):
        steps = [{"suffix_idx": j, "op": s["op"], "args": s["args"], "predicted": s["response"],
                  "ground_truth": s["response"], "value_match": True, "status_match": True, "well_formed": True}
                 for j, s in enumerate(it["trace_counterfactual"][:40])]
        return {"item_id": it["item_id"], "domain": it["domain"], "intervention_kind": it["intervention"]["kind"],
                "prefix_len": it["prefix_len"], "suffix_len": len(steps), "n_well_formed": len(steps),
                "n_value_match": len(steps), "n_status_match": len(steps), "first_value_mismatch_at": None,
                "expected_delay": None, "billing": {"n_calls": len(steps)}, "phase_a": steps,
                "phase_b": None, "phase_c": None, E.ROW_CONFIG_KEY: self.config,
                E.ROW_ID_KEY: self.rp.catalog_row_id(uid_of or it)}

    def _main(self, *extra):
        argv = ["eval_llm_counterfactual.py", "--cf-jsonl", str(self.cf), "--model", "test/model", "--strategy",
                "sliding", "--window", "20", "--max-items", "24", "--suffix-budget", "40", "--no-explain",
                "--no-state", "--prompt-version", "v1", "--out", str(self.out), *extra]
        buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"litellm": _offline_litellm()}), \
                contextlib.redirect_stdout(buf):
            E.main()
        return buf.getvalue()

    def _write(self, rows):
        self.ckpt.write_text("\n".join([json.dumps({E.CHECKPOINT_CONFIG_KEY: self.config})] +
                                       [json.dumps(r) for r in rows]) + "\n")

    def test_adv24_clean_resume_keeps_every_variant(self):
        from grader import grade_value_match
        self.assertEqual((len(self.items), len({i["item_id"] for i in self.items})), (24, 10))
        self._write([self._oracle(it) for it in self.items])
        log = self._main()
        self.assertIn("0 items pending", log)
        out = json.loads(self.out.read_text())
        self.assertFalse(out["mixed_run_configurations"])
        hits = n = 0
        for row, it in zip(out["per_item"], self.items):
            self.assertEqual(row[E.ROW_ID_KEY], self.rp.catalog_row_id(it))
            for st in row["phase_a"]:
                n += 1
                hits += bool(grade_value_match(st["predicted"], it["trace_counterfactual"][st["suffix_idx"]]["response"])["match"])
        self.assertEqual((hits, n), (960, 960))

    def test_row_without_identity_is_not_reused(self):
        rows = [self._oracle(it) for it in self.items]
        for r in rows:
            del r[E.ROW_ID_KEY]
        self._write(rows)
        with self.assertRaises(SystemExit) as cm:
            self._main()
        self.assertIn("carry no catalog-row identity", str(cm.exception))

    def test_row_answering_another_variant_is_rejected_even_with_override(self):
        a, b = next((i, j) for i in range(24) for j in range(24)
                    if i != j and self.items[i]["item_id"] == self.items[j]["item_id"])
        rows = [self._oracle(it) for it in self.items]
        rows[a] = self._oracle(self.items[b], uid_of=self.items[a])   # b's content under a's identity
        self._write(rows)
        for extra in ((), ("--allow-checkpoint-mismatch",)):
            with self.assertRaises(SystemExit) as cm:
                self._main(*extra)
            self.assertIn("do not answer their catalog rows", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
