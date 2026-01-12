#!/usr/bin/env bash
set -euo pipefail

BASE_URL=${1:-http://localhost:1234/v1}
MODEL=${2:-qwen/qwen3-vl-4b}

echo "[P1] Checking LM Studio server at ${BASE_URL}"
curl -s "${BASE_URL}/models" | head -c 2000
echo
echo "[P1] Sending a simple structured JSON prompt..."
curl -s "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [{"role":"user","content":"Return a JSON object with keys \"a\" and \"b\" = 1 and 2. Output JSON only."}],
    "temperature": 0.0,
    "max_tokens": 128,
    "stream": false
  }' | head -c 2000
echo
