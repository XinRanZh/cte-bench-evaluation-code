"""Prompt versions: v1.1 is pinned by digest and differs from v1 only in its
three documented changes.

Prompts are built as `eval_item` builds them (original source attached, line
numbers normalized) for every Core-v1 and no-effect-v1 scenario, every protocol,
and the 40 scored future calls along the oracle path (61,200 prompts). With
CTE_REFERENCE_RESULTS set, the check is repeated along the released free-rollout
predictions of the four reference models, with the harness's free-rollout
history rule. The prompt-v1 digest test is in test_no_effect_v1.py.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import core_v1  # noqa: E402
import eval_llm_counterfactual as E  # noqa: E402

BUDGET = 40
WINDOW = 20
STRATEGIES = ("sliding", "prefix_only", "free_rollout")
ROWS = core_v1.load_core_v1() + core_v1.load_no_effect_v1()

# Inverse of the three v1.1 changes (used only to check that nothing else differs).
_RE_HDR = re.compile(r"^INTERVENTION \(in effect for every call after the first (\d+) observed calls\):$", re.M)
_RE_TAG = re.compile(r"^CALL (\d+) \[(?:original program|after intervention)\]: ", re.M)
_RE_STATE_TAIL = re.compile(r"mutated between step (\d+) and step (\d+)\.")


def v1_1_to_v1(p: str, item: dict) -> str:
    if item["intervention"].get("kind") in ("state_injection", "clock_jump"):
        p = _RE_HDR.sub("INTERVENTION (applied at step k in the trace):", p, count=1)
        p = _RE_STATE_TAIL.sub("mutated between step k-1 and step k.", p, count=1)
    else:
        p = _RE_HDR.sub(f"INTERVENTION (applied at step {item['intervention'].get('line_number', '?')}):", p, count=1)
    p = p.replace(E.HISTORY_HEADER_V1_1, E.HISTORY_HEADER_V1, 1)
    return _RE_TAG.sub(lambda m: f"CALL {m.group(1)}: ", p)


def prepared(row: dict) -> dict:
    item = copy.deepcopy(row)
    E._attach_original_source(item)
    return E.normalize_line_number_convention(item)


def prompt_pairs(predictions=None):
    """Yield (item, v1 prompt, v1.1 prompt). predictions: item_id -> list of 40
    free-rollout predictions (None = unparseable); default is the oracle path."""
    for row in ROWS:
        item = prepared(row)
        strategies = STRATEGIES if predictions is None else ("free_rollout",)
        if predictions is not None and item["item_id"] not in predictions:
            continue
        for strategy in strategies:
            hist = []
            for j, s in enumerate(item["trace_counterfactual"][:BUDGET]):
                yield (item,
                       E.make_counterfactual_prompt(item, hist, s["op"], s["args"], strategy, WINDOW),
                       E.make_counterfactual_prompt_v1_1(item, hist, s["op"], s["args"], strategy, WINDOW))
                if predictions is None:
                    resp = s["response"]
                else:
                    pred = predictions[item["item_id"]][j]
                    resp = pred if pred is not None else {"error": {"code": "MODEL_INVALID_JSON"}}
                hist.append({"op": s["op"], "args": s["args"], "response": resp})


class TestPromptV1_1(unittest.TestCase):
    V1_1_DIGEST = "3e5116c314bdf0ff0da1ad3dc2ff4b920cd2a057ebe235a93a250473a21a74de"

    def test_v1_1_prompt_digest(self):
        h = hashlib.sha256()
        n = 0
        for _, _, p11 in prompt_pairs():
            h.update(p11.encode())
            n += 1
        self.assertEqual(n, 61200)
        self.assertEqual(h.hexdigest(), self.V1_1_DIGEST)

    def _check_pairs(self, pairs) -> int:
        n = 0
        for item, p1, p11 in pairs:
            n += 1
            self.assertNotEqual(p1, p11)
            self.assertEqual(v1_1_to_v1(p11, item), p1, item["item_id"])
            # (a)/(b): one header line naming k = prefix_len
            self.assertEqual(_RE_HDR.findall(p11), [str(item["prefix_len"])])
            # (c): every listed history call carries a label, and only those lines changed
            calls_v1 = len(re.findall(r"^CALL \d+: ", p1, re.M))
            self.assertEqual(len(_RE_TAG.findall(p11)), calls_v1)
            changed = [a for a, b in zip(p1.splitlines(), p11.splitlines()) if a != b]
            self.assertEqual(len(p1.splitlines()), len(p11.splitlines()))
            state = item["intervention"].get("kind") in ("state_injection", "clock_jump")
            self.assertEqual(len(changed), 2 + calls_v1 + (1 if state else 0))
        return n

    def test_v1_1_differs_from_v1_only_in_the_three_documented_places(self):
        self.assertEqual(self._check_pairs(prompt_pairs()), 61200)

    def test_dispatch_and_defaults(self):
        item = prepared(ROWS[0])
        s = item["trace_counterfactual"][0]
        self.assertEqual(E.build_counterfactual_prompt(item, [], s["op"], s["args"], "sliding", WINDOW, "v1"),
                         E.make_counterfactual_prompt(item, [], s["op"], s["args"], "sliding", WINDOW))
        self.assertEqual(E.build_counterfactual_prompt(item, [], s["op"], s["args"], "sliding", WINDOW),
                         E.make_counterfactual_prompt_v1_1(item, [], s["op"], s["args"], "sliding", WINDOW))
        with self.assertRaises(ValueError):
            E.build_counterfactual_prompt(item, [], s["op"], s["args"], "sliding", WINDOW, "v2")
        self.assertEqual(inspect.signature(E.eval_item).parameters["prompt_version"].default, "v1.1")
        out = subprocess.run([sys.executable, "scripts/eval_llm_counterfactual.py", "--help"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("--prompt-version {v1,v1.1}", out)

    def test_v1_template_unchanged(self):
        item = prepared(ROWS[0])
        s = item["trace_counterfactual"][0]
        p1 = E.make_counterfactual_prompt(item, [], s["op"], s["args"], "sliding", WINDOW)
        self.assertIn(E.HISTORY_HEADER_V1 + "\n", p1)
        self.assertIn("INTERVENTION (applied at step ", p1)


REF = os.environ.get("CTE_REFERENCE_RESULTS")


@unittest.skipUnless(REF, "set CTE_REFERENCE_RESULTS to the HF reference_results/ directory")
class TestPromptV1_1AlongReleasedRollouts(unittest.TestCase):
    def test_inverse_along_free_rollout_predictions(self):
        total = 0
        for name in ("deepseek_v4_flash", "kimi_k25", "qwen36_35b_a3b", "claude_sonnet46"):
            run = json.loads((Path(REF) / "raw" / f"{name}_free_rollout.json").read_text())
            preds = {}
            for row in run["per_item"]:
                steps = sorted(row["phase_a"], key=lambda s: int(s["suffix_idx"]))[:BUDGET]
                preds[row["item_id"]] = [s.get("predicted") for s in steps]
            with self.subTest(model=name):
                n = TestPromptV1_1._check_pairs(self, prompt_pairs(preds))
                self.assertEqual(n, 255 * BUDGET)
                total += n
        self.assertEqual(total, 40800)


if __name__ == "__main__":
    unittest.main()
