
#!/usr/bin/env bash
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=${OPENAI_API_BASE:-http://10.0.0.63:1234/v1}
export OPENAI_API_KEY=${OPENAI_API_KEY:-lm-studio}
export LMSTUDIO_MODEL=${LMSTUDIO_MODEL:-qwen/qwen3-vl-4b}
export MAX_THOUGHT_TOKENS=${MAX_THOUGHT_TOKENS:-512}
export WANDB_PROJECT=${WANDB_PROJECT:-specsloeval}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
