
NTFY_URL="https://ntfy.sh/spliemagicexp"


# Test 1 
# CON_LIST=(1 2 4 8 16 32)
# PAYLOAD_LIST=(64 256 1024)

# TRIALS=5
# WARMUP=10
# REPEATS=100
# TIMEOUT_SEC=300

# Test 2 
# CON_LIST=(1 2 4 8)
# CON_LIST=(16 32)
CON_LIST=(64)
PAYLOAD_LIST=(4096 16384 32768 65536 102400)

TRIALS=5
WARMUP=10
REPEATS=100
TIMEOUT_SEC=4500

# Authenticate before the experiments
sudo -v || exit 1

# Keep sudo authentication alive during the sweep
(
    while sleep 60; do
        sudo -n -v || break
    done
) &

SUDO_KEEPALIVE_PID=$!
trap 'kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true' EXIT

SUCCESS_COUNT=0
FAIL_COUNT=0

TOTAL_START_MS=$(date +%s%3N)

mkdir -p results/physical/logs
mkdir -p results/physical/raw
mkdir -p results/physical/configs

# Generate a unique experiment ID
RUN_ID=$(date +%Y%m%d_%H%M%S_%N)

CONFIG_FILE="results/physical/configs/sweep_${RUN_ID}.txt"

# Save experiment configuration
cat > "$CONFIG_FILE" <<EOF
run_id=$RUN_ID
con_list=${CON_LIST[*]}
payload_list_kib=${PAYLOAD_LIST[*]}
trials=$TRIALS
warmup=$WARMUP
repeats=$REPEATS
timeout_sec=$TIMEOUT_SEC
namespace=gpucon-sender
receiver=192.168.250.1:5002
EOF

echo "Configuration saved: $CONFIG_FILE"


# Repeat experiment 
for ((TRIAL=1; TRIAL<=TRIALS; TRIAL++)); do
    for CON in "${CON_LIST[@]}"; do 
        for PAYLOAD in "${PAYLOAD_LIST[@]}"; do 
        
            echo "Starting experiment: CON=$CON Payload=${PAYLOAD}KiB Trial=$TRIAL"

            START_MS=$(date +%s%3N)

            LOG_FILE="results/physical/logs/${RUN_ID}_con${CON}_payload${PAYLOAD}KiB_trial${TRIAL}.log"

            CSV_FILE="results/physical/raw/${RUN_ID}_con${CON}_payload${PAYLOAD}KiB_trial${TRIAL}.csv"

            # exec command in a specific network namespace
            if sudo ip netns exec gpucon-sender sudo -u syu23 \
                timeout --kill-after=5s "${TIMEOUT_SEC}s" ./gpuconbench sender \
                --con "$CON" \
                --payload-kib "$PAYLOAD" \
                --warmup "$WARMUP" \
                --repeats "$REPEATS" \
                --output "$CSV_FILE" > "$LOG_FILE" 2>&1

            then 
                EXIT_CODE=0
            
            else 
                EXIT_CODE=$?
            fi

            cat "$LOG_FILE"

            if [ "$EXIT_CODE" -eq 0 ]; then 
                echo "SUCCESS: CON=$CON"
                SUCCESS_COUNT=$((SUCCESS_COUNT + 1 ))
            else 
                echo "FAILED: CON=$CON (exit code=$EXIT_CODE)"
                FAIL_COUNT=$((FAIL_COUNT + 1 ))
            fi

            END_MS=$(date +%s%3N)

            ELAPSED_MS=$((END_MS - START_MS))

            if [ "$EXIT_CODE" -eq 0 ]; then 
                MESSAGE="SUCCESS: CON=$CON | Payload=${PAYLOAD}KiB | Trial=$TRIAL| Elapsed=${ELAPSED_MS} ms"
            else
                
                if [ "$EXIT_CODE" -eq 124 ]; then
                    REASON="Timeout: exceeded ${TIMEOUT_SEC} seconds"

                elif [ "$EXIT_CODE" -eq 137 ]; then
                    REASON=$(grep -im 1 -E \
                        'out of memory|oom|cannot allocate memory' \
                        "$LOG_FILE" || true)

                    if [ -z "$REASON" ]; then
                        REASON="SIGKILL (exit 137): OOM not confirmed"
                    fi

                else
                    REASON=$(grep -im 1 -E \
                        'out of memory|oom|connection refused|timed out|timeout|^Error:' \
                        "$LOG_FILE" || true)

                    if [ -z "$REASON" ]; then
                        REASON="Unknown error (exit code=$EXIT_CODE)"
                    fi
                fi

                MESSAGE="FAILED: CON=$CON | Payload=${PAYLOAD}KiB | Trial=$TRIAL |  Elapsed=${ELAPSED_MS} ms | Reason=$REASON"
            fi
            # Send notification
            # Notify only when an experiment fails
            if [ "$EXIT_CODE" -ne 0 ]; then
                if curl -fsS --max-time 10 \
                    -H "Title: GPUConBench FAILED" \
                    -d "$MESSAGE" \
                    "$NTFY_URL" > /dev/null
                then
                    echo "Failure notification sent."
                else
                    echo "WARNING: ntfy notification failed."
                fi
            fi

            echo "Elapsed time: ${ELAPSED_MS} ms"
            echo "------------------------------"
        done
    done 
done 

TOTAL_END_MS=$(date +%s%3N)
TOTAL_ELAPSED_MS=$((TOTAL_END_MS - TOTAL_START_MS))
TOTAL_COUNT=$((SUCCESS_COUNT + FAIL_COUNT))

SUMMARY="GPUConBench Finished
Total: $TOTAL_COUNT
Success: $SUCCESS_COUNT
Failed: $FAIL_COUNT
Elapsed: ${TOTAL_ELAPSED_MS} ms"

echo "$SUMMARY"

# Always send final summary
if curl -fsS --max-time 10 \
    -H "Title: GPUConBench Sweep Finished" \
    -d "$SUMMARY" \
    "$NTFY_URL" > /dev/null
then
    echo "Final notification sent."
else
    echo "WARNING: final ntfy notification failed."
fi

echo "All experiments finished."