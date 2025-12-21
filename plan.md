# Plan — P1 SpecSLOEval Reproducibility (Inference-Only)

Make Paper 1 reproducible and arXiv-ready by tightening evaluation, decoding modes, metrics, and W&B lineage. This plan is inference-only and reuses existing tasks; no downloads or training.

## Architecture sketch
criteria.yaml -> eval/CLI -> rollout engine -> providers -> validator/retry -> metrics (accuracy/faithfulness/stability/latency) -> W&B artifacts + paper tables/figures

## Requirements
- Inference-only; no RL training, no dataset downloads.
- Adopt `configs/criteria/p1_core_public_v2.yaml` as canonical criteria; keep root `criteria.yaml` as a copy for compatibility and preserve the prior version as `criteria.p1.v0.yaml`.
- Criteria schema loader with strict validation and deterministic criteria hash.
- Decoding modes U/P/P+V/S across providers, with retries and self-consistency counted in latency.
- End-to-end latency instrumentation (request start/end, provider calls, validation windows).
- Deterministic judge-based faithfulness with atomic statements and contradiction rate.
- Stability metrics with canonical JSON rules, Disagreement@k, TotalAgreementRate@k.
- Online W&B logging enforced for P1; log criteria/tasks/schemas/episodes/summary artifacts and per-episode table.
- Paper table/figure scripts that consume artifacts and write to `papers/p1/`.
- Allow larger output budgets (local LLMs): increase max tokens defaults; no `max_tokens=-1`.

## Scope
- In: `agent_stable_slo/config/criteria_v1.py`, `agent_stable_slo/eval/faithfulness_judge.py`, updates to providers/engine/stability/logging, new scripts under `scripts/paper/`, criteria YAML(s), docs.
- Out: training loops, RL sweeps, new task generation, external downloads.

## GRPO alignment (P2 follow-up)
- Keep P1 inference-only; do not change GRPO training logic in this phase.
- After P1 is stable, align GRPO eval/logging to read the same criteria schema, criteria hash, and artifact lineage used in P1.
- Ensure GRPO evaluation uses the same metric vector and gating thresholds for apples-to-apples comparisons.

## Current anchors (repo)
- Suite tasks: `tasks/clinc_en.jsonl`, `tasks/hotpot_dev.jsonl`, `tasks/t3_tools.jsonl` (see `docs/t_suite.md`).
- Schemas: `tasks/schemas/clinc_nlu_schema.json`, `tasks/schemas/hotpot_explainer_schema.json`, `tasks/schemas/t3_tool_call_schema.json`.
- Rollout: `agent_stable_slo/rollout/engine.py` + providers.
- Stability: `agent_stable_slo/eval/stability_harness.py`.
- Logging: `agent_stable_slo/logging/wandb_utils.py`.

## Action items
[ ] Promote criteria to `configs/criteria/p1_core_public_v2.yaml`, preserve `criteria.p1.v0.yaml`, and keep root `criteria.yaml` synced.
[ ] Add strict criteria schema loader (Pydantic, extra=forbid, deterministic hash, `--criteria` required).
[ ] Implement decoding modes U/P/P+V/S in providers and `rollout/engine.py`; include validator+retry and self-consistency.
[ ] Add latency instrumentation fields (request/provider/validation timestamps, retry_count, tokens_in/out).
[ ] Implement judge-based faithfulness scorer module and integrate into eval; log sampled judge traces as artifacts.
[ ] Tighten stability harness with canonicalization and deterministic seeds; add Disagreement@k and TotalAgreementRate@k.
[ ] Enforce online W&B logging and add artifact lineage + per-episode W&B Table.
[ ] Add `scripts/paper/p1_make_tables.py` and `scripts/paper/p1_make_figures.py` to emit fixed outputs for LaTeX.
[ ] Update docs with runnable smoke tests for local/remote endpoints (use positive max_tokens).

## Minimal diff strategy
- Add new modules and scripts first.
- Patch existing modules in small, reversible steps (engine, providers, stability, wandb utils, CLI).
- Keep existing CLI flags working; add new flags behind explicit `--criteria`.

## Testing and validation
- Smoke tests:
  - `curl http://localhost:1234/v1/chat/completions` and remote endpoint with valid JSON payloads.
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
