# Plan — P1 SpecSLOEval Reproducibility (Inference-Only)

Make Paper 1 reproducible and arXiv-ready by tightening evaluation, decoding modes, metrics, and W&B lineage. This plan is inference-only and reuses existing tasks; no downloads or training.

## Architecture sketch
criteria.yaml -> eval/CLI -> rollout engine -> providers -> validator/retry -> metrics (accuracy/faithfulness/stability/latency) -> W&B artifacts + paper tables/figures

## Requirements
- Inference-only; no RL training.
- Adopt `configs/criteria/p1_core_public_v2.yaml` as canonical criteria; keep root `criteria.yaml` as a copy for compatibility and preserve the prior version as `criteria.p1.v0.yaml`.
- Criteria schema loader with strict validation and deterministic criteria hash.
- Decoding modes U/P/P+V/S across providers, with retries and self-consistency counted in latency.
- End-to-end latency instrumentation (request start/end, provider calls, validation windows).
- Deterministic judge-based faithfulness with atomic statements and contradiction rate.
- Stability metrics with canonical JSON rules, Disagreement@k, TotalAgreementRate@k.
- Online W&B logging enforced for P1; log criteria/tasks/schemas/episodes/summary artifacts and per-episode table.
- Paper table/figure scripts that consume artifacts and write to `papers/p1/`.
- Allow larger output budgets (local LLMs): increase max tokens defaults; no `max_tokens=-1`.
- Model sweep for P1: `qwen/qwen3-vl-4b`, `openai/gpt-oss-20b`, `mistralai/ministral-3-3b` with cross-judging.
- A2 ladder: explicit rungs for repair and self-consistency, plus SLO budget sweep to surface the knee.

## Scope
- In: `agent_stable_slo/config/criteria_v1.py`, `agent_stable_slo/eval/faithfulness_judge.py`, updates to providers/engine/stability/logging, new scripts under `scripts/paper/`, criteria YAML(s), docs.
- Out: training loops, RL sweeps, new task generation, external downloads.

## GRPO alignment (P2 follow-up)
- Keep P1 inference-only; do not change GRPO training logic in this phase.
- After P1 is stable, align GRPO eval/logging to read the same criteria schema, criteria hash, and artifact lineage used in P1.
- Ensure GRPO evaluation uses the same metric vector and gating thresholds for apples-to-apples comparisons.

## Current anchors (repo)
- Suite tasks: `tasks/clinc_en.jsonl` (500), `tasks/hotpot_dev.jsonl` (1000), `tasks/t3_tools.jsonl` (500).
- Schemas: `tasks/schemas/clinc_nlu_schema.json`, `tasks/schemas/hotpot_explainer_schema.json`, `tasks/schemas/t3_tool_call_schema.json`.
- Rollout: `agent_stable_slo/rollout/engine.py` + providers.
- Stability: `agent_stable_slo/eval/stability_harness.py`.
- Logging: `agent_stable_slo/logging/wandb_utils.py`.

## Action items
[x] Promote criteria to `configs/criteria/p1_core_public_v2.yaml`, preserve `criteria.p1.v0.yaml`, and keep root `criteria.yaml` synced.
[x] Add strict criteria schema loader (Pydantic, extra=forbid, deterministic hash, `--criteria` required).
[x] Implement decoding modes U/P/P+V/S in providers and `rollout/engine.py`; include validator+retry and self-consistency.
[x] Add latency instrumentation fields (request/provider/validation timestamps, retry_count, tokens_in/out).
[x] Implement judge-based faithfulness scorer module and integrate into eval; log sampled judge traces as artifacts.
[x] Tighten stability harness with canonicalization and deterministic seeds; add Disagreement@k and TotalAgreementRate@k.
[x] Enforce online W&B logging and add artifact lineage + per-episode W&B Table.
[x] Add `scripts/paper/p1_make_tables.py` and `scripts/paper/p1_make_figures.py` to emit fixed outputs for LaTeX.
[x] Fix CLINC150 label mapping and schema; add label-type guard + unit test.
[x] Expand T2 (Hotpot) to >=1000 via deterministic prompt variants.
[x] Expand T3 tools to >=500 via templated generation.
[x] Add explicit A2 rungs: SPEC_DRIVEN_PLUS_REPAIR and SPEC_DRIVEN_PLUS_SELFCONSISTENCY.
[x] Log retry/repair/candidate counts and latency boundary metadata in summaries.
[x] Wire vLLM adapter with structured output version gating.
[x] Add LM Studio smoke script with endpoint override (localhost/10.0.0.63/10.0.0.72).
[~] Run extensive smoke tests (3 models, all rungs) and full Qwen3 local + remote.
    - Smoke done; full runs timed out with stability_k=5. Use --stability-k 1 for full sweeps, then run a separate stability sweep at k=5.
[ ] Update docs with runnable smoke tests for local/remote endpoints (use positive max_tokens).

## Minimal diff strategy
- Add new modules and scripts first.
- Patch existing modules in small, reversible steps (engine, providers, stability, wandb utils, CLI).
- Keep existing CLI flags working; add new flags behind explicit `--criteria`.

## Testing and validation
- Smoke tests:
  - `bash scripts/p1_smoke_lmstudio.sh http://localhost:1234/v1 qwen/qwen3-vl-4b`
  - `bash scripts/p1_smoke_lmstudio.sh http://10.0.0.72:1234/v1 openai/gpt-oss-20b`
  - `pytest -q tests/test_clinc_labels.py`
- End-to-end:
  - `python -m agent_stable_slo.cli eval --criteria configs/criteria/p1_v1.yaml --suite p1_core --mode SPEC_DRIVEN`
  - `python scripts/paper/p1_make_tables.py` and `python scripts/paper/p1_make_figures.py`
- Quality gates: `ruff check .`, `black --check .`, `pytest -q`.

## Risks and edge cases
- Provider differences in structured output APIs; fallback paths required.
- Task/schema path mismatches; criteria must match existing suite files.
- W&B auth failures; enforce online mode with clear error.
- Larger token budgets may hurt SLOs; keep budgets configurable.

## Rollback
- Revert commit or remove new files; no data migration required.
