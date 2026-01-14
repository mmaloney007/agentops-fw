# T-Suite: Tiered Evaluation Benchmarks

Deterministic task sets for SLO-aware agent evaluation, progressing from simple classification to end-to-end software engineering.

## Tier Overview

| Tier | Benchmark | Tasks | What It Tests | Industry Standard |
|------|-----------|-------|---------------|-------------------|
| **T1** | CLINC-150 | 500 | Intent classification + OOS detection | Yes (Larson et al. 2019) |
| **T2** | HotpotQA | 1000 | Multi-hop grounded reasoning | Yes (Yang et al. 2018) |
| **T3** | Tool Calls | 500 | Single tool selection + arguments | Custom |
| **T4** | BFCL v4 | 500 | Multi-function calling (industry standard) | Yes (Berkeley 2024) |
| **T5** | SWE-bench Lite | 300 | End-to-end issue resolution | Yes (Princeton 2024) |

## Task Details

### T1: Intent Classification (CLINC-150)
- **Source:** `tasks/clinc_en.jsonl`
- **Schema:** `tasks/schemas/clinc_nlu_schema.json`
- **Fields:** `intent`, `domain`, `is_oos`
- **Evaluation:** Exact match on intent, OOS F1

### T2: Grounded Reasoning (HotpotQA)
- **Source:** `tasks/hotpot_dev.jsonl`
- **Schema:** Requires answer + supporting rationale
- **Evaluation:** Answer F1, evidence recall

### T3: Tool Selection (Custom)
- **Source:** `tasks/t3_tools.jsonl`
- **Schema:** `tasks/schemas/t3_tool_call_schema.json`
- **Tools:** `lookup_customer`, `fetch_metric`, `open_incident`, `summarize_report`
- **Evaluation:** Tool match, argument accuracy

### T4: Function Calling (BFCL v4)
- **Source:** `tasks/t4_bfcl.jsonl`
- **Schema:** `tasks/schemas/t4_function_call_schema.json`
- **Categories:** `simple_python` (400), `multiple` (100)
- **Paper:** [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)
- **Evaluation:** AST-based function call matching

### T5: Software Engineering (SWE-bench Lite)
- **Source:** `tasks/t5_swebench.jsonl`
- **Schema:** `tasks/schemas/t5_patch_schema.json`
- **Repos:** Django (114), SymPy (77), Matplotlib (23), scikit-learn (23), etc.
- **Paper:** [SWE-bench: Can Language Models Resolve Real-world Github Issues?](https://www.swebench.com/)
- **Evaluation:** Test pass rate (FAIL_TO_PASS tests)

## Regenerating Tasks

```bash
# T1: CLINC-150 (intent classification)
python scripts/build_clinc_tasks.py --count 500 --out tasks/clinc_en.jsonl

# T2: HotpotQA (grounded reasoning)
python scripts/build_hotpot_tasks.py --count 1000 --out tasks/hotpot_dev.jsonl

# T3: Tool calls (custom)
python scripts/build_t_tasks.py

# T4: BFCL v4 (function calling) - requires: pip install bfcl-eval
python scripts/build_t4_bfcl.py --count 500 --categories simple_python multiple

# T5: SWE-bench Lite (software engineering) - requires: pip install datasets
python scripts/build_t5_swebench.py --count 300
```

## Running Evaluation

```bash
# Single tier
python scripts/eval_t_suite.py \
  --models lmstudio:qwen/qwen3-4b-thinking-2507 \
  --tasks tasks/t4_bfcl.jsonl \
  --mode structured \
  --out-dir out/evals

# Full suite (T1-T5)
python scripts/eval_t_suite.py \
  --models lmstudio:qwen/qwen3-4b-thinking-2507 ollama:llama3.1:8b \
  --tasks tasks/clinc_en.jsonl tasks/hotpot_dev.jsonl tasks/t3_tools.jsonl tasks/t4_bfcl.jsonl tasks/t5_swebench.jsonl \
  --mode structured \
  --out-dir out/evals
```

Set `AOFW_PROVIDER`/`LMSTUDIO_MODEL`/`OLLAMA_MODEL`/`VLLM_MODEL` as needed.

## Why This Progression?

The T1→T5 progression tests increasingly complex agent capabilities:

1. **T1 (NLU):** Can the agent understand user intent?
2. **T2 (Reasoning):** Can it reason over evidence?
3. **T3 (Tools):** Can it select and parameterize tools?
4. **T4 (Orchestration):** Can it handle industry-standard function calling?
5. **T5 (End-to-end):** Can it ship working code?

This progression directly supports the thesis: **SLO-aware training improves agent reliability across all tiers, from simple classification to complex software engineering tasks.**
