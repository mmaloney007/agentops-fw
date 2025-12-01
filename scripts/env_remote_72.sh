#!/usr/bin/env bash
# LM Studio endpoint on 10.0.0.72 running openai/gpt-oss-20b (OpenAI-compatible)
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=http://10.0.0.72:1234/v1
export OPENAI_API_KEY=lm-studio
export LMSTUDIO_MODEL="openai/gpt-oss-20b"
export MAX_THOUGHT_TOKENS=${MAX_THOUGHT_TOKENS:-196}
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_ENTITY=${WANDB_ENTITY:-mike007}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
