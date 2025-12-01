
#!/usr/bin/env bash
# Codex shell profile: ensure (micro)mamba is initialized, then activate the chosen env,
# then export repo vars for LM Studio @ 10.0.0.63.
# Point your tool to run: bash -lc "source scripts/codex_profile.sh && <your command>"
set -e

# Init (micro)mamba
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  ACT="micromamba"
elif command -v mamba >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  ACT="mamba"
else
  echo "[warn] (micro)mamba not found; proceeding without env activation."
  ACT=""
fi

# Prefer named env if present, fall back to base
if [ -n "$ACT" ]; then
  if $ACT env list | grep -E '^[^#]*agent-slo' >/dev/null 2>&1; then
    $ACT activate agent-slo
  else
    $ACT activate base || true
  fi
fi

# Repo + provider
export AOFW_PROVIDER=${AOFW_PROVIDER:-lmstudio}
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://10.0.0.63:1234/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-lm-studio}
export LMSTUDIO_MODEL=${LMSTUDIO_MODEL:-qwen/qwen3-4b-thinking-2507}
export MAX_THOUGHT_TOKENS=${MAX_THOUGHT_TOKENS:-196}

# Observability (defaults to personal W&B entity)
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_ENTITY=${WANDB_ENTITY:-mike007}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
# export WEAVE_PROJECT=agent-stable-slo

# Hand control back to the caller's command
