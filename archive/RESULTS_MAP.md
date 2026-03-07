# Results Organization Map

**Last Updated**: 2026-01-23
**Purpose**: Track where all experimental results are stored and how they map to paper content.

---

## Directory Structure Overview

```
agentops-fw/
├── out/                          # Raw experimental outputs
│   ├── p1_comprehensive_20260118/  # P1 evaluation results (13 models × 5 tasks)
│   ├── p1_eval_20260118/           # Earlier P1 run (subset)
│   └── p2_training*/               # P2 training runs (multiple directories)
├── results/                      # Aggregated summaries
│   ├── p2_training_summary.json    # 66 training runs aggregated
│   └── p2_training_results.csv     # Same data in CSV format
├── papers/                       # Paper sources and compiled PDFs
│   ├── P1_stable_slo/arxiv/        # Paper 1 LaTeX
│   ├── P2_reward_stability/arxiv/  # Paper 2 LaTeX
│   ├── P1_deployment_gap.pdf       # Compiled P1 (9 pages)
│   └── P2_capacity_thresholds.pdf  # Compiled P2 (10 pages)
└── PROGRESS.md                   # Human-readable progress dashboard
```

---

## P1 Evaluation Results

### Primary Data Location
**File**: `out/p1_comprehensive_20260118/all_results.json`

### Contents
- **13 models** evaluated across **5 task types**
- **2,300 total examples** (500 per task for T1-T4, 300 for T5)
- Per-model, per-task metrics:
  - `avg_latency_ms`, `p95_latency_ms`, `p99_latency_ms`
  - `latency_slo_pct` (% meeting SLO deadline)
  - `success_at_slo_pct` (% correct AND on time)
  - `json_valid` (structural validity rate)
  - Task-specific accuracy metrics

### Models Evaluated
| # | Model | Size | Vendor | Success@SLO |
|---|-------|------|--------|-------------|
| 1 | llama-3.2-1b | 1B | Meta | 49.1% |
| 2 | llama-3.2-3b | 3B | Meta | 31.8% |
| 3 | qwen2.5-3b | 3B | Alibaba | 34.3% |
| 4 | phi-3-mini | 3.8B | Microsoft | 33.8% |
| 5 | qwen3-4b | 4B | Alibaba | 29.9% |
| 6 | yi-1.5-6b | 6B | 01.AI | 0.4% |
| 7 | mistral-7b-v0.3 | 7B | Mistral | 10.0% |
| 8 | falcon-mamba-7b | 7B | TII | 0.0% |
| 9 | gpt-oss-20b | 20B | OpenAI | 0.6% |
| 10 | ministral-8b | 8B | Mistral | 28.5% |
| 11 | llama-3.1-8b | 8B | Meta | 13.2% |
| 12 | gemma-2-9b | 9B | Google | 2.2% |
| 13 | gemma-3-12b | 12B | Google | 0.0% |

### Task Types
| Task | Description | SLO | Examples |
|------|-------------|-----|----------|
| T1 | CLINC-150 intent classification | 2000ms | 500 |
| T2 | HotpotQA grounded QA | 2000ms | 500 |
| T3 | Tool calling | 2000ms | 500 |
| T4 | BFCL function routing | 2000ms | 500 |
| T5 | SWE-bench code patching | 10000ms | 300 |

### Maps to Paper 1
- **Table 1** (main results): Success@SLO, P95 latency
- **Table 2** (per-task breakdown): T1-T5 Success@SLO by model
- **Figure 1** (deployment gap scatter): Accuracy vs Success@SLO
- **Figure 3** (rank inversion): Accuracy rank vs SLO rank

---

## P2 Training Results

### Primary Data Location
**File**: `results/p2_training_summary.json`
**CSV**: `results/p2_training_results.csv`

### Contents
- **66 training runs** (11 models × 3 seeds × 2 step counts)
- **2 models blocked** by hardware (Falcon-Mamba-7B, GPT-OSS-20B)
- Per-run metrics:
  - `valid_pct` (overall JSON validity)
  - `last50_valid_pct` (final 50 steps validity - key metric)
  - `avg_reward` (composite reward)
  - `total_steps` (training steps completed)

### Training Configurations
- **Seeds**: 42, 123, 456
- **Steps**: 250, 500
- **Algorithm**: GRPO with LoRA (rank=16, alpha=32)
- **Hardware**: RTX 4090 (24GB), 4-bit quantization

### Key Findings
| Model | Size | Avg Last-50% | Learning? |
|-------|------|--------------|-----------|
| Gemma-3-12B | 12B | **79%** | Yes |
| Gemma-2-9B | 9B | **53%** | Yes |
| Qwen2.5-3B | 3B | 17% | Outlier |
| Yi-1.5-6B | 6B | 9% | Weak |
| All others | <8B | 0-5% | No |

### Raw Training Output Directories
```
out/p2_training/                    # Main training runs
├── gemma-2-9b_seed*_*steps/
├── gemma-3-12b_seed*_*steps/
├── llama-3.2-1b_seed*_*steps/
├── llama-3.2-3b_seed*_*steps/
├── llama-3.1-8b_seed*_*steps/
├── ministral-8b_seed*_*steps/
├── mistral-7b_seed*_*steps/
├── qwen2.5-3b_seed*_*steps/
└── qwen3-4b_seed*_*steps/

out/p2_training_fixed/              # Re-runs with fixes
├── phi-3-mini_seed*_*steps/
└── yi-1.5-6b_seed*_*steps/
```

### Maps to Paper 2
- **Table 1** (main results): Final-50 validity, avg reward, trend
- **Figure 1** (learning curves): JSON validity over training steps
- **Figure 3** (capacity threshold): Model size vs learning ability

---

## Hardware Blockers

### Falcon-Mamba-7B
- **Issue**: CUDA version mismatch prevents mamba-ssm fast kernels
- **Impact**: "slow_forward" path OOMs on 24GB VRAM
- **Requirement**: >24GB VRAM or matching CUDA versions

### GPT-OSS-20B
- **Issue**: Mxfp4 dequantization OOMs during model load
- **Impact**: Cannot start training
- **Requirement**: Estimated 32-40GB VRAM

---

## Paper Compilation

### Paper 1: The Deployment Gap
- **Source**: `papers/P1_stable_slo/arxiv/main.tex`
- **PDF**: `papers/P1_deployment_gap.pdf`
- **Pages**: 9
- **Status**: Complete with all 13 models × 5 tasks

### Paper 2: Capacity Thresholds
- **Source**: `papers/P2_reward_stability/arxiv/main.tex`
- **PDF**: `papers/P2_capacity_thresholds.pdf`
- **Pages**: 10
- **Status**: Complete with representative models (Qwen3-4B, Gemma-3-12B)

### Compile Commands
```bash
cd papers/P1_stable_slo/arxiv && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
cd papers/P2_reward_stability/arxiv && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Future Work Data Needs

### P2 Extended Training (proposed)
- **Goal**: 2000+ steps to confirm sustained learning
- **Models**: Gemma-2-9B, Gemma-3-12B
- **Output**: `out/p2_extended/`

### Qwen2.5-3B Investigation (proposed)
- **Goal**: 10 seeds to determine if outlier is real
- **Model**: Qwen2.5-3B only
- **Output**: `out/p2_qwen_investigation/`

### PPO Comparison (proposed)
- **Goal**: Compare GRPO vs PPO on one model
- **Model**: Qwen3-4B (small, below threshold)
- **Output**: `out/p2_ppo_comparison/`

---

## Quick Access Commands

```bash
# View P1 results summary
python -c "import json; d=json.load(open('out/p1_comprehensive_20260118/all_results.json')); print(json.dumps({m: d['models'][m]['tasks'] for m in list(d['models'].keys())[:3]}, indent=2))"

# View P2 training summary
python -c "import json; print(json.dumps(json.load(open('results/p2_training_summary.json')), indent=2))"

# Check progress dashboard
cat PROGRESS.md | head -60
```
