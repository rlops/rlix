#!/bin/bash
# Watchdog wrapper: launches a smoke script and kills it if the log goes
# silent for SILENCE_LIMIT seconds, or if total runtime exceeds RUN_LIMIT
# seconds. Usage: SCRIPT=/path/to/smoke.sh bash run_smoke_with_watchdog.sh

SCRIPT=${SCRIPT:-/root/rlix/scripts/run_smoke_dual.sh}
SILENCE_LIMIT=${SILENCE_LIMIT:-300}    # 5 min between log lines
RUN_LIMIT=${RUN_LIMIT:-2400}            # 40 min total
LOG=${LOG:-/root/logs/run.log}

mkdir -p $(dirname $LOG)
: > $LOG

echo "=== watchdog starting smoke: $SCRIPT ===" | tee -a $LOG
bash $SCRIPT >$LOG 2>&1 &
RUN_PID=$!

START=$(date +%s)
LAST_SIZE=0
LAST_CHANGE=$START

while kill -0 $RUN_PID 2>/dev/null; do
    NOW=$(date +%s)
    SIZE=$(stat -c%s $LOG 2>/dev/null || echo 0)
    if [ "$SIZE" != "$LAST_SIZE" ]; then
        LAST_SIZE=$SIZE
        LAST_CHANGE=$NOW
    fi
    SILENT=$((NOW - LAST_CHANGE))
    ELAPSED=$((NOW - START))
    if [ "$SILENT" -gt "$SILENCE_LIMIT" ]; then
        echo "=== WATCHDOG: log silent for $SILENT seconds, killing run ===" >> $LOG
        kill -9 $RUN_PID 2>/dev/null
        pkill -9 -f run_miles_dual 2>/dev/null
        pkill -9 -f run_miles_rlix 2>/dev/null
        pkill -9 -f MegatronTrain 2>/dev/null
        pkill -9 -f sglang 2>/dev/null
        ray stop --force >/dev/null 2>&1
        echo EXIT_CODE=124 >> $LOG  # timeout exit code
        exit 124
    fi
    if [ "$ELAPSED" -gt "$RUN_LIMIT" ]; then
        echo "=== WATCHDOG: total runtime $ELAPSED > $RUN_LIMIT, killing run ===" >> $LOG
        kill -9 $RUN_PID 2>/dev/null
        pkill -9 -f run_miles_dual 2>/dev/null
        pkill -9 -f run_miles_rlix 2>/dev/null
        pkill -9 -f MegatronTrain 2>/dev/null
        pkill -9 -f sglang 2>/dev/null
        ray stop --force >/dev/null 2>&1
        echo EXIT_CODE=125 >> $LOG  # cap-hit exit code
        exit 125
    fi
    sleep 5
done

wait $RUN_PID
RC=$?
echo EXIT_CODE=$RC >> $LOG
exit $RC
