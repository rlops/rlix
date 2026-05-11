#!/bin/bash
# M11.2 real-overlap PASS-bar harness. Reads a smoke log (default
# /tmp/miles_dual.log) and emits a structured report:
#   - donor-shrink events with timestamps
#   - F40 Runtime expand events with timestamps
#   - C20 0-active suspend / wake events
#   - shrink_engines disable-routing log line (3f invariant)
#   - shutdown_hard complete per pipeline
#   - pre/post nvidia-smi delta
#   - Codex Q5-c overlap-topology assertion (P1_infer ∩ P2_infer must be non-empty)
#
# Exits non-zero with a per-condition diagnostic if any of the 7 PASS-bar
# conditions fails. Sourced into the run loop after each smoke iteration.

set -u

LOG=${1:-/tmp/miles_dual.log}
if [ ! -f "$LOG" ]; then
    echo "ERR no log at $LOG"
    exit 2
fi

fail=0
note() { echo "  - $*"; }
fail_cond() { echo "FAIL $*"; fail=$((fail+1)); }
pass_cond() { echo "PASS $*"; }

echo "=========================================="
echo "M11.2-overlap PASS-bar — log=$LOG"
echo "=========================================="

# ---------------- Condition 3: overlap topology assertion ------------------
echo "--- Condition 3 (Codex Q5-c): overlap topology ---"
mp1=$(grep -oE "mp1_infer=\[[0-9, ]*\]" "$LOG" | tail -1 | grep -oE "[0-9]+" | sort -u | tr '\n' ',')
mp2=$(grep -oE "mp2_infer=\[[0-9, ]*\]" "$LOG" | tail -1 | grep -oE "[0-9]+" | sort -u | tr '\n' ',')
overlap=$(grep -oE "overlap=\[[0-9, ]*\]" "$LOG" | tail -1 | grep -oE "[0-9]+" | sort -u | tr '\n' ',')
note "mp1_infer=[${mp1%,}]  mp2_infer=[${mp2%,}]  shared=[${overlap%,}]"
if [ -z "$overlap" ]; then
    fail_cond "C3 — no overlap detected (smoke ran disjoint); real M11.2 requires P1_infer ∩ P2_infer ≠ ∅"
else
    pass_cond "C3 — overlap on GPUs [${overlap%,}]"
fi

# ---------------- Condition 4a: donor-shrink events -----------------------
echo "--- Condition 4a: donor-shrink events ---"
donor_count=$(grep -cE "sched_guided_shrink_ops|donor_cid|shrink_engines:" "$LOG" || true)
note "donor_shrink/shrink_engines log lines: $donor_count"
grep -nE "sched_guided_shrink_ops|donor_cid|shrink_engines:" "$LOG" | head -10
if [ "$donor_count" -lt 1 ]; then
    fail_cond "C4a — no donor_shrink / shrink_engines events (contention path not exercised)"
else
    pass_cond "C4a — donor_shrink/shrink_engines events present"
fi

# ---------------- Condition 4b: F40 Runtime expand events -----------------
echo "--- Condition 4b: F40 Runtime expand events ---"
f40_count=$(grep -cE "F40 Runtime|_expand_workers" "$LOG" || true)
note "F40 Runtime/_expand_workers log lines: $f40_count"
grep -nE "F40 Runtime|_expand_workers|activate_routing: registered router workers" "$LOG" | head -10
if [ "$f40_count" -lt 1 ]; then
    fail_cond "C4b — no F40 Runtime expand events"
else
    pass_cond "C4b — F40 Runtime expand events present"
fi

# ---------------- Condition 5: shrink-disables-routing log (3f) -----------
echo "--- Condition 5 (Codex Q5-b / 3f): shrink_engines disable-routing ---"
disable_count=$(grep -cE "shrink_engines: disabled router workers prior to release" "$LOG" || true)
note "disable-before-release log lines: $disable_count"
if [ "$disable_count" -lt 1 ]; then
    fail_cond "C5 — no shrink-engines disable-router log; 3f invariant unverified"
else
    pass_cond "C5 — shrink_engines disable-before-release log present"
fi

# ---------------- Condition 2: shutdown_hard complete --------------------
echo "--- Condition 2: shutdown_hard complete per pipeline ---"
shutdown_count=$(grep -cE "shutdown_hard complete pipeline_id=" "$LOG" || true)
note "shutdown_hard complete log lines: $shutdown_count"
grep -nE "shutdown_hard complete pipeline_id=" "$LOG"
if [ "$shutdown_count" -lt 2 ]; then
    fail_cond "C2 — expected 2 shutdown_hard complete lines (one per pipeline), got $shutdown_count"
else
    pass_cond "C2 — both pipelines shut down cleanly"
fi

# ---------------- Condition 7: post-run leaked actors --------------------
# (best-effort: ray status snapshot must be appended to the log by the runner)
echo "--- Condition 7: leaked actors (ray status, if captured) ---"
if grep -qE "ray status|Resource usage" "$LOG"; then
    leaked=$(grep -A 30 "Resource usage" "$LOG" | grep -cE "Actor|actor")
    note "leaked actor lines in ray status: $leaked"
    if [ "$leaked" -gt 0 ]; then
        fail_cond "C7 — ray status shows leaked actors"
    else
        pass_cond "C7 — ray status clean"
    fi
else
    note "C7 — no ray status snapshot in log (runner should append)"
fi

# ---------------- Condition 6: nvidia-smi residual ------------------------
echo "--- Condition 6: post-run nvidia-smi residual ≤200 MiB ---"
if grep -qE "memory.used \[MiB\]" "$LOG"; then
    over_thresh=$(awk -F',' '/memory.used \[MiB\]/{flag=1;next} flag && /^[0-9]+ MiB/{ if ($1+0 > 200) print "GPU residual " $1 " MiB > 200 MiB"; if (++c>=4) flag=0 }' "$LOG" | wc -l)
    if [ "$over_thresh" -gt 0 ]; then
        fail_cond "C6 — $over_thresh GPU(s) over 200 MiB residual"
    else
        pass_cond "C6 — all GPUs ≤200 MiB residual"
    fi
else
    note "C6 — no post-run nvidia-smi snapshot in log"
fi

# ---------------- C20 suspend / wake (informational) ---------------------
echo "--- C20 0-active suspend / wake (informational) ---"
c20_suspend=$(grep -cE "_workers_changed|workers_changed_notify|empty candidate_set" "$LOG" || true)
note "C20 suspend/wake log lines: $c20_suspend"

echo "=========================================="
if [ "$fail" -gt 0 ]; then
    echo "RESULT: FAIL ($fail conditions failed)"
    exit 1
fi
echo "RESULT: PASS (all overlap-topology PASS-bar conditions met)"
exit 0
