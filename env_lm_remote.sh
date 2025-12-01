#!/usr/bin/env bash
# Convenience env for LM Studio running remotely at 10.0.0.63:1234
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://10.0.0.63:1234/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-lm-studio}
export LMSTUDIO_MODEL=${LMSTUDIO_MODEL:-qwen/qwen3-4b-thinking-2507}
export AOFW_STREAM=1
export MAX_THOUGHT_TOKENS=${MAX_THOUGHT_TOKENS:-196}
# Observability (set these if not already set in your shell)
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_ENTITY=${WANDB_ENTITY:-mike007}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
# Optional Weave
# export WEAVE_PROJECT=agent-stable-slo
