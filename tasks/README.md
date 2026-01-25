# Task Suite

Evaluation and training tasks for SpecSLOEval.

## Core Tasks (T1-T5)

Used in Papers P1 and P2. These are the canonical task files.

| ID | File | N | Description | SLO | Metric |
|----|------|---|-------------|-----|--------|
| T1 | `clinc_en.jsonl` | 500 | Intent classification (CLINC-150, 150 classes) | 2s | `t1_field_acc` |
| T2 | `hotpot_dev.jsonl` | 1000 | Multi-hop QA (HotpotQA) | 2s | `t2_summary_f1` |
| T3 | `t3_tools.jsonl` | 500 | Tool calling with typed arguments | 2s | `t3_success` |
| T4 | `t4_bfcl.jsonl` | 500 | Function routing (Berkeley FCL) | 2s | `t4_func_match` |
| T5 | `t5_swebench.jsonl` | 300 | Code patching (SWE-bench lite) | 10s | `t5_has_patch` |

## Training Mixes

| File | N | Description |
|------|---|-------------|
| `t1t5_balanced.jsonl` | 500 | Equal mix: 100 per task |
| `t1t5_natural.jsonl` | 1416 | Natural distribution |

## Public Benchmarks (Available)

Not yet used in papers. Could extend evaluation.

| File | N | Source |
|------|---|--------|
| `public_gsm8k.jsonl` | 200 | Math word problems |
| `public_humaneval.jsonl` | 164 | Code completion |
| `public_mbpp.jsonl` | 200 | Python programming |
| `public_truthfulqa.jsonl` | 200 | Factual QA |

## Legacy (Do Not Use)

Superseded by core tasks above.

| File | Notes |
|------|-------|
| `t1_structured.jsonl` | 10-example prototype |
| `t2_grounded.jsonl` | 6-example prototype |
| `t2_expanded.jsonl` | Intermediate version |
| `robust_eval*.jsonl` | Development only |
| `faithfulness_*.jsonl` | Development only |
| `tiny_smoke.jsonl` | 1-example smoke test |

## Format

All task files use JSONL format:
```json
{"id": "unique_id", "prompt": "...", "schema_path": "tasks/schemas/X.json", "gold": {...}}
```

Schemas are in `schemas/`.
