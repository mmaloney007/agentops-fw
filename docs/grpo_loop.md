# GRPO loop hardening

Key features added to the GRPO trainer/eval scripts:
- Pydantic-validated config (`GRPOTrainConfig`) with explicit defaults, versioning, config presets (`configs/grpo/*.yaml`), and a `--config-file` loader with “no silent defaults” guard.
- Repro switches: `--seed`, `--repro/--deterministic` set seeds across Python/NumPy/torch, enable deterministic algorithms, and shard indices per rank (`RANK`/`WORLD_SIZE`).
- Dataset integrity: fingerprinting + optional cached copy (`--cache-dataset --cache-dir out/cache`) that rewrites schema paths for stable runs; optional `--expected-dataset-hash` enforcement and drift warnings.
- Run manifest at `<out>/manifest.json` capturing config, dataset fingerprints (source + cached), hardware info, git hash/status, env snapshot, and distro rank info.
- Structured logs at `<out>/run_log.jsonl` plus step-level metrics in `<out>/train_log.jsonl` (encode/gen/backward/optim timings, throughput, GPU mem) with bootstrap CIs in summaries.
- Checkpoints: `--checkpoint-every N` writes adapter + optimizer/scaler/RNG state under `<out>/checkpoints/`; resume with `--resume-from <ckpt or run dir>`.
- Validation hook: optional `--val-tasks/--val-interval/--val-samples` to run best-of-k validation during training.
- Safety guard: optional `--max-prompt-chars` + `--truncate-prompts` to cap incoming prompts.
- Safer training: NaN/inf loss guard, deterministic generation when requested, GradScaler on CUDA, W&B logging gated by `WANDB_PROJECT`, prompt/schema validation on load.
- Blocklist guardrails: `--blocklist` + `--reject-blocklisted` to zero reward when blocked substrings appear.

Typical training run:
```
python -m agent_stable_slo.train.grpo_train_loop \
  --config-file configs/grpo/4090_qwen3.yaml \
  --tasks tasks/robust_eval_gold.jsonl \
  --out out/grpo_qwen3_$(date +%s) \
  --steps 500 --max-new-tokens 128 \
  --seed 1234 --repro \
  --cache-dataset --checkpoint-every 100 \
  --val-tasks tasks/robust_eval_gold.jsonl --val-interval 100 --val-samples 1
```

Other presets (swap base models; LoRA + 4bit already configured):
- Llama 3.1 8B Instruct (4090): `--config-file configs/grpo/4090_llama3.yaml`
- Gemma 2 9B Instruct (4090): `--config-file configs/grpo/4090_gemma2.yaml`

Eval loop (best-of-k sampling):
```
python -m agent_stable_slo.train.grpo_trl \
  --tasks tasks/fc_tasks.jsonl --steps 200 --samples 2 \
  --seed 1234 --repro --cache-dataset
```

Smoke tests (uses a tiny HF model):
```
pytest tests/test_grpo_train_loop_smoke.py::test_grpo_train_loop_runs_one_step_tiny_model -q
# or 300-step smoke with tiny data/model:
python -m agent_stable_slo.train.grpo_train_loop \
  --config-file configs/grpo/tiny_smoke.yaml \
  --tasks tasks/tiny_smoke.jsonl \
  --steps 300 --eval-interval 50 --max-new-tokens 32 --cache-dataset
```

Checkpoint sweeps & artifacts:
- Evaluate all checkpoints under a run: `python scripts/eval_checkpoints.py --run-dir out/<run> --tasks tasks/robust_eval_gold.jsonl --steps 100 --samples 1`
- Package a run dir with checksums (and optional W&B upload): `python scripts/package_artifacts.py --run-dir out/<run> --archive`
- Summarize logs into tables: `python scripts/summarize_logs.py --inputs out/<run>/train_log.jsonl --out table.csv`

Distributed smoke (single node):
- `torchrun --nproc_per_node=1 -m agent_stable_slo.train.grpo_train_loop --config-file configs/grpo/tiny_smoke.yaml --tasks tasks/tiny_smoke.jsonl --steps 10 --eval-interval 5`
- For >1 rank, set `--ddp-backend nccl|gloo` and run with `torchrun --nproc_per_node=<N> ...` (ensure per-rank HF cache or shared FS).

Notebook for approachable P1 evals/tables: see `notebooks/p1_eval.ipynb` (defaults to tiny model/tasks; swap in your Qwen/RL runs and paper tasks).

Output layout:
- `out/<run>/manifest.json`: config, dataset fingerprints, hardware/env snapshot, git hash/status.
- `out/<run>/run_log.jsonl`: structured info/summary events (includes validation steps if enabled).
- `out/<run>/train_log.jsonl` or `eval.jsonl`: per-step metrics/outputs.
- `out/<run>/checkpoints/step_XXXXXX/`: adapter, tokenizer, optimizer/scaler, RNG state (when checkpointing enabled).
