
#!/usr/bin/env bash
export AOFW_PROVIDER=lmstudio
export OPENAI_API_BASE=http://10.0.0.63:1234/v1
export OPENAI_API_KEY=lm-studio
export LMSTUDIO_MODEL="Qwen2.5-7B-Instruct"
export MAX_THOUGHT_TOKENS=196
export WANDB_PROJECT=${WANDB_PROJECT:-agent-stable-slo}
export WANDB_DIR=${WANDB_DIR:-$(pwd)/wandb_logs}
