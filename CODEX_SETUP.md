
# Codex + (Micro)Mamba integration

You can run Codex (or any automation that shells into your repo) with your **mamba base** profile
or a dedicated env (recommended: `agent-slo`).

## Option A — Use the **base** profile
1. Ensure `(micro)mamba` is installed and base is initialized.
2. Point your tool to run commands with the provided profile prelude:
   ```bash
   bash -lc "source scripts/codex_profile.sh && <your command>"
   ```
   `scripts/codex_profile.sh` activates `agent-slo` if it exists, otherwise `base`, and exports
   LM Studio + W&B env variables (10.0.0.72 + Qwen).

## Option B — Use a dedicated env (`agent-slo`) [recommended]
1. Create env from YAML (once):
   ```bash
   micromamba create -f environment.yml -n agent-slo -y
   # or: mamba env create -f environment.yml
   ```
2. Install Torch **per OS** after activation:
   - macOS: `pip install torch==2.4.1`
   - Ubuntu + CUDA 12.1:
     ```bash
     pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
     ```
3. In automations, call:
   ```bash
   bash -lc "source scripts/codex_profile.sh && <your command>"
   ```

## Quick smoke tests
- Check LM Studio model list:
  ```bash
  curl -s http://10.0.0.72:1234/v1/models | jq -r '.data[].id' | head
  ```
- Baseline one‑liner:
  ```bash
  bash -lc "source scripts/codex_profile.sh && python -m agent_stable_slo.train.grpo_trl --tasks tasks/fc_tasks.jsonl --out out/codex_smoke --steps 50 --max-new-tokens 64"
  ```

## Notes
- Using **base** works, but a dedicated env avoids dependency drift.
- If W&B is offline, set `WANDB_MODE=offline` and sync later with `wandb sync`.
