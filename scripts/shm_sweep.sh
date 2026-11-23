#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BIN="./build/shmbench"
SHM="/dev/shm/gpuconbench-shm-test"
SOCK="/tmp/gpuconbench-shm-test.sock"

# Physical Ethernet Small experiment baseline
sizes=("4MiB" "16MiB" "32MiB" "64MiB" "100MiB")
cons=(1 2 4 8 )

warmup=10
repeats=100
trials=5

LOG_DIR="results/shm/sweep_logs/$(date +%Y%m%d_%H%M%S_%N)"
mkdir -p "$LOG_DIR"

SWEEP_SUMMARY="$LOG_DIR/sweep_summary.csv"

printf '%s\n' \
    'trial_id,payload_size,concurrency,warmup,repeats,total_requests,batch_elapsed_ms,aggregate_throughput_mib_s' \
    > "$SWEEP_SUMMARY"

receiver_pid=""

cleanup() {
    if [[ -n "$receiver_pid" ]] && kill -0 "$receiver_pid" 2>/dev/null; then
        kill "$receiver_pid" 2>/dev/null || true
        wait "$receiver_pid" 2>/dev/null || true
    fi
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ ! -x "$BIN" ]]; then
    echo "Binary not found. Run: make shm"
    exit 1
fi

for size in "${sizes[@]}"; do
    for con in "${cons[@]}"; do

        echo "================================"
        echo "Payload=$size CON=$con"
        echo "================================"

        # Avoid conflicting with an existing Receiver
        if [[ -e "$SHM" || -e "$SOCK" ]]; then
            echo "Existing SHM or socket detected."
            echo "Check whether another Receiver is running."
            exit 1
        fi

        args=(
            --size "$size"
            --con "$con"
            --warmup "$warmup"
            --repeats "$repeats"
            --trials "$trials"
        )

        receiver_log="$LOG_DIR/receiver_${size}_con${con}.log"
        sender_log="$LOG_DIR/sender_${size}_con${con}.log"

        # Start Receiver
        "$BIN" receiver "${args[@]}" > "$receiver_log" 2>&1 &
        receiver_pid=$!

        # Wait until the Unix socket is available
        ready=0

        for ((i=0; i<100; i++)); do
            if [[ -S "$SOCK" ]]; then
                ready=1
                break
            fi

            if ! kill -0 "$receiver_pid" 2>/dev/null; then
                break
            fi

            sleep 0.1
        done

        if (( ready == 0 )); then
            echo "Receiver startup failed."
            cat "$receiver_log"
            exit 1
        fi

        # Run Sender
        if ! "$BIN" sender "${args[@]}" > "$sender_log" 2>&1; then
            echo "Sender failed."
            cat "$sender_log"
            exit 1
        fi

        # Check that all trials completed
        expected=$((con * repeats))
        completed=$(grep -Fc "Measurements collected: $expected" "$sender_log" || true)

        if [[ "$completed" -ne "$trials" ]] ||
           ! grep -q "Run Summary CSV saved:" "$sender_log"; then
            echo "Sender did not complete all trials."
            cat "$sender_log"
            exit 1
        fi

        # Wait for Receiver to exit
        if ! wait "$receiver_pid"; then
            echo "Receiver failed."
            cat "$receiver_log"
            exit 1
        fi
        receiver_pid=""

        if grep -q "Receiver worker error:" "$receiver_log"; then
            echo "Receiver reported an error."
            cat "$receiver_log"
            exit 1
        fi

        # Find the Run Summary created by this Sender
        run_summary_path=$(
            sed -n 's/^Run Summary CSV saved: //p' "$sender_log" |
            tail -n 1
        )

        if [[ -z "$run_summary_path" || ! -f "$run_summary_path" ]]; then
            echo "Run Summary CSV not found."
            exit 1
        fi

        # Validate trial count
        record_count=$(awk 'END {print (NR > 0 ? NR-1 : 0)}' "$run_summary_path")

        if [[ "$record_count" -ne "$trials" ]]; then
            echo "Unexpected trial count: $record_count (expected $trials)"
            exit 1
        fi

        # Append records without the CSV header
        tail -n +2 "$run_summary_path" >> "$SWEEP_SUMMARY"

        # Display brief results
        grep -E \
            "^(Batch Elapsed:|Aggregate Throughput:|Run Summary CSV saved:)" \
            "$sender_log"

        echo "Completed: $size CON=$con"
        echo
    done
done

echo "All SHM sweep combinations completed."
echo "Logs: $LOG_DIR"
echo "Sweep Summary CSV saved: $SWEEP_SUMMARY"
