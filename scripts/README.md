# Scripts

Main offline commands:

```bash
python -m py_compile scripts/*.py testbeds/*.py
python scripts/leaderboard.py --result-jsons reference_results/raw/*.json --out-dir scratch/leaderboard
```

Main online evaluation command:

```bash
python scripts/eval_llm_counterfactual.py \
  --cf-jsonl counterfactuals/core_v1.jsonl \
  --model deepseek/deepseek-chat \
  --strategy sliding --window 20 \
  --max-items 255 --suffix-budget 40 \
  --out scratch/model_result.json
```

Protocol names:

- `--strategy sliding --window 20`: TF-1step sliding K=20.
- `--strategy prefix_only`: no-suffix-feedback.
- `--strategy free_rollout --window 20`: self-conditioned rollout.
