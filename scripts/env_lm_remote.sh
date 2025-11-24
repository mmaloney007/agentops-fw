#!/usr/bin/env bash
# Convenience env for LM Studio running remotely at 10.0.0.72:1234
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=http://10.0.0.72:1234/v1
export OPENAI_API_KEY=lm-studio
export LMSTUDIO_MODEL="Qwen2.5-7B-Instruct"
export AOFW_STREAM=1
export MAX_THOUGHT_TOKENS=196
# Observability (set these if not already set in your shell)
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
# Optional Weave
# export WEAVE_PROJECT=agent-stable-slo
