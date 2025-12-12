## P1 augmentation notes (Dec 2025)

Use these to drop the current numbers into the paper and caveat eval-time latencies.

### Canonical run (serving metrics)
- Source: `out/p1_paper1_content_2000_wb` (W&B run: https://wandb.ai/mike007/agent_paper_1/runs/g7n23g5t).
- Metrics (`train_summary.csv`): reward_mean 1.6048 (CI 1.55–1.65), json_valid_rate 0.997, latency p95 3099 ms (p99 3136 ms), ttft_avg 1882 ms, on 4090 BF16, 2k steps, max_prompt_len 1152, max_new_tokens 128, CONTENT_WEIGHT=1.0, T=0.9, top_p=0.92, no 4-bit.
- Plot: `reward_latency.png` shows stable latency around ~3s and reward rising with occasional dips; use in the Experiments section.
- Recommended wording: “On a single RTX 4090, our spec-driven GRPO run achieves ~99.7% JSON validity and p95 ≈3.1s (p99 ≈3.14s) on long contexts, with composite reward_mean 1.60.”

### Faithfulness slices (appendix/qualitative)
- Mini (5 ex): reward_mean 2.80; plus (10 ex): reward_mean 2.80. Files: `eval_faithfulness.json/.csv`, `eval_faithfulness_plus.json/.csv`; tasks: `faithfulness_mini.jsonl`, `faithfulness_plus.jsonl`.
- W&B: mini https://wandb.ai/mike007/agent_paper_1/runs/0hu1ar3i; plus https://wandb.ai/mike007/agent_paper_1/runs/7904cn0e.
- Caveat: eval p95 latencies are high (~25–28s) because each ckpt eval reloads base+adapter; treat as content/faithfulness signal only, not serving latency.
- Suggested wording: “On a 10-example grounded set, we obtain faithfulness-weighted reward ≈2.8 with 100% JSON validity; eval latency is dominated by per-ckpt model load and is not representative of serving latency.”

### Checkpoint sweep
- Single-ckpt sweep: `eval_ckpts.json/.csv` (ckpt step_001999 reward_mean 2.56; eval p95 ~20.8s due to model reload). Mention only if needed; clarify it is eval-time overhead.

### File drop locations
- `papers/P1_stable_slo/artifacts/`: train_summary.csv, table.csv, reward_latency.png, eval_ckpts.json/.csv, eval_faithfulness*.json/.csv, tasks/faithfulness_*.jsonl.

### Paper integration checklist
- Add a short paragraph in Experiments citing the serving metrics above (JSON validity, reward_mean, p95) and the hardware/config.
- Add one figure (reward_latency.png) and one table row from train_summary.csv into the main text; move faithfulness numbers to appendix with the latency caveat.
- Explicitly note eval-vs-serving latency distinction for checkpoint/faithfulness sweeps.
- Tie back to claims: contract-first + SLO on single GPU; content-aware reward is active; faithfulness slice shows directionality.***
