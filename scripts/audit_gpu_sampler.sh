#!/bin/bash
# GPU memory sampler for the M11 offload audit.
# Emits one block per second to $OUT:
#   T=<epoch> GPU <idx>,<used_mib>; ...            (whole-GPU memory.used)
#   T=<epoch> APP gpu=<idx> pid=<pid> mem_mib=<m> cmd=<argv0..>   (per compute process)
OUT=${OUT:-/root/logs/gpu_samples.log}
INTERVAL=${INTERVAL:-1}

# bus_id -> index map (bus ids in compute-apps output)
declare -A BUS2IDX
while IFS=, read -r idx bus; do
  bus=$(echo "$bus" | tr -d ' ')
  idx=$(echo "$idx" | tr -d ' ')
  BUS2IDX[$bus]=$idx
done < <(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader)

while true; do
  ts=$(date +%s)
  {
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ';' | tr -d ' ')
    echo "T=$ts GPU $g"
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,used_memory --format=csv,noheader,nounits |
    while IFS=, read -r bus pid mem; do
      bus=$(echo "$bus" | tr -d ' '); pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | cut -c1-160)
      echo "T=$ts APP gpu=${BUS2IDX[$bus]:-$bus} pid=$pid mem_mib=$mem cmd=$cmd"
    done
  } >> "$OUT"
  sleep "$INTERVAL"
done
