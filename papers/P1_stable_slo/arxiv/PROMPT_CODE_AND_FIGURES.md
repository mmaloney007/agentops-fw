## 0) Context

I run single-node local inference on:
- Ubuntu 22.04 + RTX 4090 (Alienware R15), and
- macOS (M2 Max).

I have **three local models** available in LM Studio right now:
- `qwen/qwen3-vl-4b`
- `openai/gpt-oss-20b`
- `mistralai/ministral-3-14b-reasoning`

I also want to compare **two backends** behind an OpenAI-compatible API:
- LM Studio server
- vLLM OpenAI API server (on the 4090)



# Prompt: code changes + paper figure generation

Copy/paste the prompt below into GPT-5 (or your preferred coding LLM) **inside your repo**.

---

You are a staff+ systems engineer and data scientist. You will modify my repo to make the paper results fully reproducible and to generate reviewer-grade figures and tables.

## Goal
Implement an **SLO-aware, spec-first evaluation harness** that:
1) runs a configuration ladder ablation (U → P → P+V → G → S → SLO-hardened),
2) sweeps **latency budgets** and (optionally) **QPS/concurrency**,
3) outputs a single canonical dataset (`runs.csv` + `requests.jsonl`) that can reproduce every plot/table in the paper,
4) generates publication-quality figures into `paper/figs/` and LaTeX tables into `paper/auto/`.

## Hard requirements
- Every request log row MUST include:
  - `run_id`, `timestamp`, `model`, `backend`, `workload`, `config_name` (U/P/P+V/G/S/SLO)
  - `seed`, `temperature`, `max_tokens`, `top_p`
  - `budget_ms` (SLO), `concurrency`, `qps_target` (or `None`)
  - `attempt_count`, `retry_policy` (e.g., none/fixed/backoff), and whether a retry occurred
  - `ttft_ms`, `latency_ms` (end-to-end), plus a breakdown: `gen_ms`, `validation_ms`, `tool_ms`, `queue_ms` (if measurable)
  - compliance booleans: `json_valid`, `schema_valid`, `tool_valid`, `citation_well_formed`
  - quality signals: `faithfulness_score` (0–1), `task_score` (if supervised), and `judge_id` + `judge_prompt_hash`
  - `output_tokens`, `input_tokens` (or best-effort approximation)
  - `error_type` (timeout/schema/tool/judge/other) and `error_message` (truncated)

- Latency metrics MUST be computed client-side at the same measurement boundary for all configs.
- Retries MUST count toward latency budget.
- The harness MUST support at least 2 models and 2 backends (e.g., LM Studio + Ollama) and must run the same workloads on all.

## Implementation tasks
### 1) Canonical config ladder
Create a `configs/` module and represent configs as data (YAML or JSON):
- `U`: unconstrained prompt-only
- `P`: provider structured output / json mode only
- `P+V`: provider + strict JSON Schema validation
- `G`: add guardrails / field-level sanitization + targeted repair prompts
- `S`: compiled constraints (grammar / constrained decoding where supported) + validation
- `SLO`: S plus SLO-hardened policies: early-exit, attempt cap, dynamic strictness, fallback format

Expose a single function:
```python
run_episode(prompt, config, budget_ms, backend, model, seed, ...) -> RequestLog
```

### 2) Workload runner
Add `scripts/run_suite.py` that runs:
- workloads from `tasks/*.jsonl` (including cited summaries)
- multiple configs
- multiple budgets (e.g., [500, 1000, 2000, 3000] ms)
- multiple seeds (>=3)
Optionally sweep concurrency levels.

The script must write:
- `artifacts/requests.jsonl` (one line per request)
- `artifacts/runs.csv` (one row per (run_id, workload, config, budget, model, backend) aggregate)

### 3) Aggregation + metrics
Add `scripts/aggregate.py` to compute for each run row:
- success metrics: `Success@SLO` (fraction of requests with `latency_ms <= budget_ms` AND all required compliance booleans true)
- latency quantiles: p50/p90/p95/p99 of `latency_ms` and `ttft_ms`
- compliance rates: JSON/schema/tool/citation validity
- failure breakdown rates by `error_type`
- judge stability: rerun judge prompt `k` times for a small stratified sample and report agreement

### 4) Figure generation
Add `scripts/make_paper_figs.py` (matplotlib only; no seaborn) that reads `runs.csv` and outputs:

**Figure 0 (Claim A1): Schema-validity != deployability**
- a simple 2-bar (or 2-point) plot for a highlighted run showing:
  - `schema_valid_rate` (or `json_valid_rate` if schema not available)
  - `Success@SLO`
- this figure must be generated directly from the aggregated `runs.csv` so the claim is mechanically reproducible.

**Figure A: Pareto frontier**
- x = p95 latency (ms) under each budget
- y = Success@SLO (or Faithfulness@SLO for summary workloads)
- one curve per config ladder stage
- facets / separate lines per model and backend

**Figure B: Ablation bars**
- schema_valid_rate and citation_well_formed_rate by config stage

**Figure C: Tail amplification from retries**
- plot attempt_count distribution and the delta between p50 and p99 by config

**Figure D: Judge credibility firewall**
- judge/human correlation on a 50-example labeled set (you can stub the label loader if labels are not present)
- judge prompt sensitivity: same examples, different judge prompt variants

Save as PDF and PNG into `paper/figs/` with deterministic filenames.

### 5) Auto-generated tables
Add `scripts/make_paper_tables.py` to output:
- `paper/auto/latency_table.tex`
- `paper/auto/failure_table.tex`
- `paper/auto/frontier_summary.tex`

Format tables with `booktabs` and **no vertical rules**.

### 6) Paper integration
Modify `paper/main.tex` (or `main.tex`) to `\input{paper/auto/latency_table.tex}` etc so figures/tables are always regenerated from source.

## Deliverables
- Full diff / patch
- New scripts with docstrings and CLI usage examples
- A one-command workflow:
  - `make eval` → runs suite
  - `make figs` → generates all paper figs/tables
  - `make paper` → builds PDF

Do not hand-wave. If any required field cannot be measured on a backend, implement a best-effort approximation and document it in-code.


## vLLM launch command (RTX 4090)

Example:

```bash
python -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 \
  --model <MODEL_NAME_OR_PATH> \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```
