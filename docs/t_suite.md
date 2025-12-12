# T1/T2/T3 task suites

Deterministic task sets for the paper:

- **T1 (CLINC NLU intent + OOS):** `intent`, `domain`, `is_oos` using `tasks/clinc_en.jsonl` and schema `tasks/schemas/clinc_nlu_schema.json`.
- **T2 (HotpotQA grounded reasoning):** answer + rationale with `tasks/hotpot_dev.jsonl` built from `hotpotqa/hotpot_qa` (distractor).
- **T3 (tool calls):** synthetic but realistic tool calls with gold arguments. Schema: `tasks/schemas/t3_tool_call_schema.json`. Data: `tasks/t3_tools.jsonl` (16 cases).
- **Optional synthetic baselines:** `tasks/t1_structured.jsonl` and `tasks/t2_grounded.jsonl` remain if you want tighter, fully in-repo cases.

Regenerate tasks (no network needed except when pulling HF datasets the first time):
```bash
python scripts/build_clinc_tasks.py --count 200 --out tasks/clinc_en.jsonl
python scripts/build_hotpot_tasks.py --count 50 --out tasks/hotpot_dev.jsonl
# synthetic structured/tool sets:
python scripts/build_t_tasks.py
# optional limits for synthetic:
# python scripts/build_t_tasks.py --t1-count 5 --t2-count 3 --t3-count 8
```

Multi-model evaluation:
```bash
python scripts/eval_t_suite.py \
  --models lmstudio:qwen/qwen3-4b-thinking-2507 ollama:llama3.1:8b \
  --tasks tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl \
  --mode structured \
  --out-dir out/evals
```
Set `AOFW_PROVIDER`/`LMSTUDIO_MODEL`/`OLLAMA_MODEL`/`VLLM_MODEL` (and base URLs) as needed. Results land under `out/evals/<run>/<provider_model>/summary.json` with per-sample logs in `predictions.jsonl`.
