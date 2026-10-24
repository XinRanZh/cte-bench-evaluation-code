# CTE-Bench Evaluation Code

This repository contains the evaluation code for CTE-Bench-Core-v1. It
contains the executable service environments, intervention/oracle code,
graders, and evaluation scripts.

Dataset artifact: <https://huggingface.co/datasets/zhangxr7/cte-bench-core-v1>

## Contents

- `testbeds/`: deterministic Python service implementations.
- `scripts/`: oracle generation, intervention, grading, and model
  evaluation scripts.
- `counterfactuals/`: Core-v1, the no-effect-v1 companion split, the
  identifier-renamed variant of the 210 source-edit scenarios, the
  earlier no-effect-v0 probe, and the delayed-effect probe scenario files
  used by the scripts.
- `tests/`: release checks for the no-effect-v1 split, the paired
  scorer, the renamed variant, and the reference policies.
- `requirements.txt`: Python dependencies for local execution.
- `LICENSE` (MIT, benchmark code), `LICENSES.md` (code vs data),
  `THIRD_PARTY_NOTICES.md` and `THIRD_PARTY_LICENSES/` (upstream copyright
  notices and full license texts for the three ported services).

## Quick Checks

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m py_compile scripts/*.py testbeds/*.py
```

## No-Effect-v1 Companion Split

Core-v1 keeps only scenarios whose intervention changes at least one
future response, so its headline score (effect-step VM) never penalizes
a model for predicting a change that does not happen.
`counterfactuals/no_effect_v1.jsonl` closes that gap: 255 no-effect
companions, one per Core-v1 scenario (`paired_item_id`), each with the
same service, the same observed prefix, and the same 40 scored future
calls as its partner, but a benign intervention whose future responses
are identical to the unpatched service at every scored position
(10,200/10,200).

Matching (see the dataset card for the full table):

- each companion is matched to its Core-v1 partner on service, observed
  prefix, and 40-call schedule, so the two splits have identical
  distributions of service, prefix length, and trajectory
  (total-variation distance 0 on each axis);
- the companion's intervention is chosen to be benign and need not have
  the partner's type (TVD 0.33 over intervention types). The 30
  clock-jump scenarios have code-patch companions, and 93/255 companions
  (`benign_tier = near_kind`) use a different intervention type than
  their partner;
- 141/255 companions are `global_no_op` (inert on every Core-v1
  trajectory of their service) and 114 are `contingent` (the same patch
  changes a response on another trajectory). Report the contingent
  subset as well; models score higher on `global_no_op` rows.

The split is generated deterministically from Core-v1 by
`scripts/build_no_effect_v1.py` (CPU only, about 15 s); rerunning it
reproduces the released file byte for byte.

## Paired Reporting Is Required

Any result on the no-effect-v1 split must be reported as the pair

    (Core-v1 effect-step VM, no-effect-v1 VM)

for the same model, protocol, and prompt, each with its scenario-level
bootstrap interval. Do not report no-effect VM alone, and do not collapse
the pair into `min()`, a mean, or a difference used as a ranking: a model
that ignores every intervention scores (0, 100), and for both reference
models `min()` is just effect-step VM. Compute the pair with:

```bash
python scripts/score_paired.py \
  --core-run path/to/core_v1_run.json \
  --no-effect-run path/to/no_effect_v1_run.json \
  --label "My model" --json scratch/paired.json
```

The scorer re-grades every step against the released catalogs, joins
the no-effect run by `item_id` and the Core-v1 run by its 40-step
schedule, refuses incomplete pairs unless `--allow-partial` is given,
and reports effect-step VM, no-effect VM, the over-prediction rate
(well-formed no-effect predictions that invent a change), and no-effect
VM on the `contingent` and `global_no_op` subsets.

To produce the no-effect run, use the same command as for Core-v1 with
`--cf-jsonl counterfactuals/no_effect_v1.jsonl` and the same
`--prompt-version` (the released reference pairs use `--prompt-version v1`).

Reference pairs for DeepSeek V4-Flash and Kimi K2.5 (answers revealed /
TF-1step sliding K=20; raw outputs in the dataset repository under
`reference_results/`):

| model | effect-step VM | no-effect VM | over-prediction | NE VM contingent / global no-op |
|---|---|---|---|---|
| DeepSeek V4-Flash | 61.2 [58.1, 64.3] | 78.8 [77.5, 80.1] | 21.2 | 75.2 / 81.7 |
| Kimi K2.5 | 61.7 [58.5, 64.7] | 75.0 [73.7, 76.3] | 24.7 | 72.8 / 76.7 |

Run metadata: Core-v1 runs April 2026, no-effect-v1 runs July 2026, same
API model identifiers and serving routes. When reporting a new model, run
both splits with the same model version, serving route, protocol, and
prompt, and state the run dates.

## Reference Policies

`scripts/reference_policies.py` scores non-simulating policies that see
exactly what a model sees under each protocol (replaying the unchanged
service; repeating the most recent visible response to the same
operation or to the same call). Read each model score next to the
reference for its protocol:

| protocol | copy last response, same call: effect-step VM |
|---|---|
| answers revealed | 19.3 |
| no feedback | 3.3 |
| free rollout | 2.2 |

```bash
python scripts/reference_policies.py --core-jsonl counterfactuals/core_v1.jsonl --json scratch/reference_policies.json
```

## Identifier-Renamed Variant

`counterfactuals/core_v1_renamed.jsonl` holds the 210 source-edit Core-v1
scenarios with the service class, operations, helper methods, and
instance attributes renamed to neutral labels by Python's tokenizer
(`paired_item_id` links each row to its Core-v1 scenario; response
values, arguments, comments, strings, and line numbers are unchanged).
`scripts/build_renamed_core.py` regenerates it byte for byte (CPU only,
about 1 s) and replays every scenario under both sources: all 13,940
observed and future responses agree. Evaluate it like Core-v1 with
`--cf-jsonl counterfactuals/core_v1_renamed.jsonl --no-explain --no-state`;
the prediction prompts show the renamed patched source, and the released
study uses `--prompt-version v1`.
`scripts/score_renamed_variant.py` scores the released same-day study
(two original and two renamed runs).

## Reasoning Off and On

`scripts/score_reasoning_pairs.py` compares DeepSeek V4-Flash with
reasoning off (`deepseek-chat`) and on (`deepseek-reasoner`, reasoning
effort high, `--max-tokens 12000`; both with `--prompt-version v1`) on
Core-v1 from the released outputs:

| protocol (scenarios) | effect-step VM off | effect-step VM on |
|---|---|---|
| answers revealed (233) | 62.3 | 81.1 |
| no feedback (255) | 24.6 | 34.5 |

## Tests

```bash
python -m unittest discover -s tests -v
# also reproduce the reference pairs from the dataset repository:
CTE_REFERENCE_RESULTS=/path/to/cte-bench-core-v1/reference_results \
  python -m unittest discover -s tests -v
```

## Offline Scoring

If you have model-output JSON files in the format produced by
`scripts/eval_llm_counterfactual.py`, run:

```bash
python scripts/leaderboard.py \
  --result-jsons path/to/model_result_1.json path/to/model_result_2.json \
  --out-dir scratch/leaderboard
```

Each input is bound to the catalog it was run on (pass `--catalog
path/to/catalog.jsonl` for a catalog outside `counterfactuals/`); an input
that answers no catalog is an error.

## Prompt Versions

- **Prompt v1.1** (recommended for new evaluations, default
  `--prompt-version v1.1`).
- **Prompt v1** (`--prompt-version v1`): the prompt of the paper's
  tables and of every released reference output. Use it to reproduce
  them.

Prompt v1.1 differs from v1 in three places:

1. the source-edit intervention header states that the intervention is
   in effect for every call after the first k observed calls
   (k = number of observed calls);
2. the state- and clock-edit header uses the same wording, and its
   boundary sentence names the numeric steps ("mutated between step {k}
   and step {k+1}" instead of "between step k-1 and step k");
3. the history header is revised, and each history call is labelled
   `[original program]` (observed call) or `[after intervention]`
   (future call).

Everything else is byte-identical (`tests/test_prompt_versions.py` checks this on
all 61,200 Core-v1 and companion prompts and along the released
free-rollout predictions). Report the prompt version with every score,
together with the model version, serving route, protocol, and inference
cost.

Checkpoints (`<out>.ckpt.jsonl`) record the run configuration (model,
protocol, window, prompt version, budgets, routing and inference
settings such as `OPENROUTER_PROVIDER`, `OPENROUTER_QUANTIZATIONS`,
`OPENROUTER_REASONING_MAX_TOKENS`, `OPENROUTER_REASONING_EFFORT`,
`AWS_REGION`, temperature and timeouts, catalog hash) in their first
line, and every row carries its `run_config` and `catalog_row_sha256`
(the identity of the catalog row it answers; item_ids can repeat, e.g. in
the adv24 probe). A resume reuses a row only for its own catalog row,
after checking its calls and targets, and only if the header and every
row match the requested configuration. `--allow-checkpoint-mismatch`
overrides the configuration check, marks the checkpoint permanently as
mixed, and the output then sets `mixed_run_configurations: true` and
lists the per-row configurations. Use a separate `--out` per prompt
version. `run_state_probe_v2.py` stamps and checks its cached records in
the same way (`--allow-mixed`).

A call that returns a provider or infrastructure error (insufficient
balance, service unavailable, timeout, auth) is recorded with
`infra_error` and no prediction. It is not a model output: every scorer
and consumer excludes it from all numerators and denominators through
one rule (`grader.is_infrastructure_failure` / `grader.scoring_view`) and
reports the excluded calls (`infrastructure_failures_excluded` in
`reference_audit.json` and in each report). A row with such a call is
evaluated again when the same command is rerun. In outputs written
before these explicit records, a call counts only if its response is the
evaluator's exception record (`ERR:` followed by the exception's dotted
class name, e.g. `ERR:litellm.Timeout: ...`) and the row's billing shows
it unbilled. Billed model output is always scored, even if its text
starts with `ERR:`, and so are billed empty responses, refusals and
truncations.

Every script that reads evaluator outputs (`score_paired.py`,
`leaderboard.py`, `audit_reference_results.py`,
`score_reasoning_pairs.py`, `score_renamed_variant.py`,
`apply_v2_filter.py`, `build_core_v1.py`, `core_v1.filter_core_results`)
goes through one validation path, `scripts/run_provenance.py`:

- an output with no provenance at all is "unstamped legacy" (all
  released reference outputs) and is scored exactly as before;
- an output with modern provenance must carry a stamp on every row; a
  partial or missing stamp is "damaged" and always rejected;
- the row stamps, the top-level `run_config` and the declared model,
  protocol, window, prompt version, reasoning, budgets and request
  settings must agree, and so must inputs combined in one report; legacy
  and stamped inputs are never combined; the two runs of a paired report
  must be one model (`run_provenance.MODEL_ALIASES`, legacy files
  included), and pair labels must be unique;
- every input, legacy or stamped, is bound to its catalog (catalog hash,
  one-to-one row join, unique in-range step indices, calls and targets
  equal to the catalog's); a row that answers no catalog row is an error.
  The leaderboard binds each input to a `--catalog` file (default: every
  catalog in `counterfactuals/`), `apply_v2_filter.py` to its `--catalog`
  or `--cf-dir` candidate pool, `build_core_v1.py` to its candidate pool,
  `core_v1.filter_core_results` to `counterfactuals/core_v1.jsonl`, and
  study scripts also check their study's settings;
- the leaderboard groups rows by the complete configuration (model,
  protocol, window, prompt version, reasoning, max_tokens, budget,
  catalog); each unstamped legacy file is its own group, labelled with its
  file name and catalog. No summary, table or plot pools different files,
  protocols or configurations.

Any disagreement is an error unless `--allow-mixed` is given; the report
then lists every configuration and carries `mixed_run_provenance` fields
and a `WARNING: MIXED RUN PROVENANCE` line. Reported settings always
come from the row stamps (or the validated declarations of legacy
files).

## Online Evaluation

Reproduce a paper run (prompt v1), e.g. TF-1step:

```bash
DEEPSEEK_API_KEY=<your-key> python scripts/eval_llm_counterfactual.py \
  --cf-jsonl counterfactuals/core_v1.jsonl \
  --model deepseek/deepseek-chat \
  --strategy sliding --window 20 \
  --max-items 255 --suffix-budget 40 \
  --prompt-version v1 \
  --out scratch/model_result_v1.json
```

Evaluate a new model with the recommended prompt v1.1 (the default):

```bash
python scripts/eval_llm_counterfactual.py \
  --cf-jsonl counterfactuals/core_v1.jsonl \
  --model <provider/model> \
  --strategy sliding --window 20 \
  --max-items 255 --suffix-budget 40 \
  --prompt-version v1.1 \
  --out scratch/model_result_v1_1.json
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
