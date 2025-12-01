
#!/usr/bin/env bash
# Usage: source scripts/activate_mamba.sh [base|agent-slo]
# Default is "agent-slo" if it exists, otherwise "base".
set -e

TARGET="${1:-}"
if ! command -v micromamba >/dev/null 2>&1 && ! command -v mamba >/dev/null 2>&1; then
  echo "[err] (micro)mamba not found. Install micromamba or mamba first." >&2
  return 1 2>/dev/null || exit 1
fi

# Shell hook
if command -v micromamba >/dev/null 2>&1; then
  eval "$(micromamba shell hook -s bash)"
  M="micromamba"
else
  eval "$(conda shell.bash hook)"
  M="mamba"
fi

# Choose env
if [ -z "${TARGET}" ]; then
  if $M env list | grep -E '^[^#]*agent-slo' >/dev/null 2>&1; then
    TARGET="agent-slo"
  else
    TARGET="base"
  fi
fi

echo "[info] activating $TARGET via $M ..."
$M activate "$TARGET"

# Repo / experiment env
export AOFW_PROVIDER=${AOFW_PROVIDER:-lmstudio}
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://10.0.0.63:1234/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-lm-studio}
export LMSTUDIO_MODEL=${LMSTUDIO_MODEL:-qwen/qwen3-4b-thinking-2507}
export MAX_THOUGHT_TOKENS=${MAX_THOUGHT_TOKENS:-196}

# W&B / Weave (default entity points to personal account; override if needed)
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_ENTITY=${WANDB_ENTITY:-mike007}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
# export WEAVE_PROJECT=agent-stable-slo

echo "[ok] env ready. Example: python -m agent_stable_slo.train.grpo_trl --help"
