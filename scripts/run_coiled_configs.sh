#!/usr/bin/env bash
set -u
set -o pipefail

RUNTIME="coiled"
LOG_DIR="run-logs/$(date +%Y%m%d_%H%M%S)"
SKIP_LLM=0
TIMEOUT_SECS=$((75 * 60))

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime)
            RUNTIME="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --skip-llm)
            SKIP_LLM=1
            shift
            ;;
        --timeout-minutes)
            TIMEOUT_SECS=$(( $2 * 60 ))
            shift 2
            ;;
        --timeout-seconds)
            TIMEOUT_SECS="$2"
            shift 2
            ;;
        --no-timeout)
            TIMEOUT_SECS=0
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

CONFIGS=(
    "configs/ecomm.yaml"
    "configs/ecomm_loyalty.yaml"
    "configs/gaming.yaml"
    "configs/media_demo_scale.yaml"
    "configs/media_small.yaml"
    "configs/media_testing_mils.yaml"
    "configs/sports_and_concert.yaml"
    "configs/wine_and_cheese.yaml"
)

mkdir -p "$LOG_DIR"
SUMMARY_CSV="$LOG_DIR/summary.csv"
SUMMARY_MD="$LOG_DIR/summary.md"

csv_escape() {
    local value="$1"
    value=${value//\"/\"\"}
    printf '"%s"' "$value"
}

md_escape() {
    local value="$1"
    value=${value//|/\\|}
    printf '%s' "$value"
}

extract_last_value() {
    local pattern="$1"
    local log_file="$2"
    if command -v rg >/dev/null 2>&1; then
        rg -N "$pattern" "$log_file" | tail -n 1
    else
        grep -E "$pattern" "$log_file" | tail -n 1
    fi
}

extract_uc_field() {
    local field="$1"
    local log_file="$2"
    awk -v field="$field" '
        /=== UC Volume ===/ {in_block=1; next}
        in_block && $0 ~ field {
            sub(/.*: /, "", $0)
            print
            exit
        }
    ' "$log_file"
}

run_with_timeout() {
    local log_file="$1"
    shift

    if [[ "$TIMEOUT_SECS" -le 0 ]]; then
        "$@" 2>&1 | tee "$log_file"
        return ${PIPESTATUS[0]}
    fi

    if command -v timeout >/dev/null 2>&1; then
        timeout "$TIMEOUT_SECS" "$@" 2>&1 | tee "$log_file"
        return ${PIPESTATUS[0]}
    fi

    if command -v gtimeout >/dev/null 2>&1; then
        gtimeout "$TIMEOUT_SECS" "$@" 2>&1 | tee "$log_file"
        return ${PIPESTATUS[0]}
    fi

    python3 -u - "$TIMEOUT_SECS" "$log_file" "$@" <<'PY'
import os
import selectors
import signal
import subprocess
import sys
import time

timeout_secs = int(sys.argv[1])
log_file = sys.argv[2]
cmd = sys.argv[3:]

start = time.monotonic()
proc = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    preexec_fn=os.setsid,
)

selector = selectors.DefaultSelector()
if proc.stdout is not None:
    selector.register(proc.stdout, selectors.EVENT_READ)

with open(log_file, "w", encoding="utf-8") as fh:
    while True:
        elapsed = time.monotonic() - start
        if elapsed > timeout_secs:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(2)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            sys.exit(124)

        if proc.poll() is not None:
            break

        events = selector.select(timeout=0.2)
        if not events:
            continue
        for key, _ in events:
            line = key.fileobj.readline()
            if not line:
                continue
            sys.stdout.write(line)
            sys.stdout.flush()
            fh.write(line)
            fh.flush()

sys.exit(proc.returncode or 0)
PY
}

echo "config,status,base_uri,uc_volume,uc_path,log_file" > "$SUMMARY_CSV"
{
    echo "| config | status | base_uri | uc_volume | uc_path | log |"
    echo "| --- | --- | --- | --- | --- | --- |"
} > "$SUMMARY_MD"

for config in "${CONFIGS[@]}"; do
    name=$(basename "$config" .yaml)
    log_file="$LOG_DIR/${name}.log"
    echo "Running $config (runtime=$RUNTIME)"

    run_env=(env -u DATABRICKS_TOKEN -u Databricks_token)
    if [[ "$SKIP_LLM" -eq 1 ]]; then
        run_env=(env -u DATABRICKS_TOKEN -u Databricks_token NL_SKIP_LLM=1)
    fi

    run_cmd=(
        pixi run neuralift_c360_prep --config "$config" --runtime "$RUNTIME"
    )
    run_with_timeout "$log_file" "${run_env[@]}" "${run_cmd[@]}"
    exit_code=$?
    if [[ "$exit_code" -eq 0 ]]; then
        status="ok"
    elif [[ "$exit_code" -eq 124 ]]; then
        status="timeout"
    else
        status="fail($exit_code)"
    fi

    base_line=$(extract_last_value "Base URI" "$log_file")
    base_uri=${base_line##*: }
    if [[ "$base_uri" == "$base_line" ]]; then
        base_uri=""
    fi
    uc_volume=$(extract_uc_field "Name" "$log_file")
    uc_path=$(extract_uc_field "UC Path" "$log_file")

    printf '%s,%s,%s,%s,%s,%s\n' \
        "$(csv_escape "$config")" \
        "$(csv_escape "$status")" \
        "$(csv_escape "$base_uri")" \
        "$(csv_escape "$uc_volume")" \
        "$(csv_escape "$uc_path")" \
        "$(csv_escape "$log_file")" \
        >> "$SUMMARY_CSV"

    printf '| %s | %s | %s | %s | %s | %s |\n' \
        "$(md_escape "$config")" \
        "$(md_escape "$status")" \
        "$(md_escape "$base_uri")" \
        "$(md_escape "$uc_volume")" \
        "$(md_escape "$uc_path")" \
        "$(md_escape "$log_file")" \
        >> "$SUMMARY_MD"
done

echo "Summary written to:"
echo "  $SUMMARY_CSV"
echo "  $SUMMARY_MD"
