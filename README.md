# CTE-Bench Evaluation Code

This repository contains the evaluation code for CTE-Bench-Core-v1. It
contains the executable service environments, intervention/oracle code,
graders, and evaluation scripts.

Dataset artifact: <https://huggingface.co/datasets/zhangxr7/cte-bench-core-v1>

## Contents

- `testbeds/`: deterministic Python service implementations.
- `scripts/`: oracle generation, intervention, grading, and model
  evaluation scripts.
- `counterfactuals/`: Core-v1, no-effect, and delayed-effect probe
  scenario files used by the scripts.
- `requirements.txt`: Python dependencies for local execution.
- `LICENSES.md` and `THIRD_PARTY_NOTICES.md`: license and provenance
  notes for synthetic and adapted services.

## Quick Checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m py_compile scripts/*.py testbeds/*.py
```

## Offline Scoring

If you have model-output JSON files in the format produced by
`scripts/eval_llm_counterfactual.py`, run:

```bash
python scripts/leaderboard.py \
  --result-jsons path/to/model_result_1.json path/to/model_result_2.json \
  --out-dir scratch/leaderboard
```

## Online Evaluation

Example command for a TF-1step run:

```bash
DEEPSEEK_API_KEY=<your-key> python scripts/eval_llm_counterfactual.py \
  --cf-jsonl counterfactuals/core_v1.jsonl \
  --model deepseek/deepseek-chat \
  --strategy sliding --window 20 \
  --max-items 255 --suffix-budget 40 \
  --out scratch/model_result.json
```

Protocol mapping:

- `--strategy sliding --window 20`: TF-1step sliding K=20.
- `--strategy prefix_only`: no-suffix-feedback.
- `--strategy free_rollout --window 20`: self-conditioned rollout.

## Notes

The code is intended for reproducible evaluation of stateful-service
simulation under fixed source/state interventions. It is not an
autonomous-agent environment: future calls are fixed by the scenario
files, and the model predicts service responses rather than selecting
actions.
