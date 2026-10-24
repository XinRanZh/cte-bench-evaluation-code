# Scripts

Main offline commands:

```bash
python -m py_compile scripts/*.py testbeds/*.py
python scripts/leaderboard.py --result-jsons reference_results/raw/*.json --out-dir scratch/leaderboard

# paired report for the no-effect-v1 companion split (required for no-effect claims)
python scripts/score_paired.py --core-run CORE_RUN.json --no-effect-run NO_EFFECT_RUN.json --label MODEL

# audit released reference outputs (coverage, duplicates, re-graded metrics)
python scripts/audit_reference_results.py --dataset-root path/to/cte-bench-core-v1 --json scratch/audit.json

# non-simulating reference policies for every protocol (CPU only)
python scripts/reference_policies.py --core-jsonl counterfactuals/core_v1.jsonl --json scratch/reference_policies.json

# identifier-renamed variant: rebuild + verify, then score the released study
python scripts/build_renamed_core.py --out scratch/core_v1_renamed.jsonl --report scratch/renamed_verification.json
python scripts/score_renamed_variant.py --dataset-root path/to/cte-bench-core-v1 --json scratch/renamed.json

# reasoning off vs on (DeepSeek V4-Flash) from the released outputs
python scripts/score_reasoning_pairs.py --dataset-root path/to/cte-bench-core-v1 --json scratch/reasoning.json

# regenerate counterfactuals/no_effect_v1.jsonl (deterministic, CPU only)
python scripts/build_no_effect_v1.py --core-jsonl counterfactuals/core_v1.jsonl \
  --out-jsonl scratch/no_effect_v1.jsonl --report scratch/no_effect_v1_build_report.md
```

Main online evaluation command (prompt v1 reproduces the paper's runs;
omit `--prompt-version` or pass `v1.1` for new evaluations):

```bash
python scripts/eval_llm_counterfactual.py \
  --cf-jsonl counterfactuals/core_v1.jsonl \
  --model deepseek/deepseek-chat \
  --strategy sliding --window 20 \
  --max-items 255 --suffix-budget 40 \
  --prompt-version v1 \
  --out scratch/model_result_v1.json
```

Protocol names:

- `--strategy sliding --window 20`: TF-1step sliding K=20.
- `--strategy prefix_only`: no-suffix-feedback.
- `--strategy free_rollout --window 20`: self-conditioned rollout.
