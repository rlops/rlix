#!/bin/bash
# Sync local rlix + miles working trees (and helper tooling) to a vast.ai instance.
#
# Usage:
#   ./scripts/sync_to_vast.sh <ssh-port> [ssh-host] [flags]
#
# Examples:
#   ./scripts/sync_to_vast.sh 18599                      # → root@ssh9.vast.ai:18599
#   ./scripts/sync_to_vast.sh 39469 ssh5.vast.ai         # other gateway
#   ./scripts/sync_to_vast.sh 18599 ssh9.vast.ai --delete --setup
#
# Flags:
#   --delete   mirror exactly (remove remote files not present locally) — use for
#              a clean state; without it rsync only adds/updates.
#   --setup    after syncing, run the one-shot environment setup on the instance
#              (deps, model, datasets, torch_dist checkpoint, SGLang patches).
#   --dry-run  show what would transfer, change nothing.
#
# What gets synced:
#   $RLIX_LOCAL  → /root/rlix    (this repo)
#   $MILES_LOCAL → /root/miles   (miles repo)
#   scripts/*.sh|*.py            → /root/   (debug/mock/setup tooling)
#
# Override paths/targets via env: RLIX_LOCAL, MILES_LOCAL, SSH_KEY,
# RLIX_REMOTE (default /root/rlix), MILES_REMOTE (default /root/miles).

set -euo pipefail

PORT="${1:?usage: sync_to_vast.sh <ssh-port> [ssh-host] [--delete] [--setup] [--dry-run]}"
shift
HOST="ssh3.vast.ai"
if [[ $# -gt 0 && "$1" != --* ]]; then HOST="$1"; shift; fi

DELETE=""; SETUP=0; DRY=""
for arg in "$@"; do
  case "$arg" in
    --delete)  DELETE="--delete" ;;
    --setup)   SETUP=1 ;;
    --dry-run) DRY="--dry-run" ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

SSH_KEY="${SSH_KEY:-$HOME/.ssh/general_private_key}"
RLIX_LOCAL="${RLIX_LOCAL:-$HOME/Library/CloudStorage/Dropbox/Python/rlix_miles}"
MILES_LOCAL="${MILES_LOCAL:-$HOME/Dropbox/Python/miles}"
RLIX_REMOTE="${RLIX_REMOTE:-/root/rlix}"
MILES_REMOTE="${MILES_REMOTE:-/root/miles}"

SSH_OPTS=(-o StrictHostKeyChecking=no -i "$SSH_KEY" -p "$PORT")
RSYNC_SSH="ssh -o StrictHostKeyChecking=no -i $SSH_KEY -p $PORT"
EXCLUDES=(--exclude .git --exclude __pycache__ --exclude '*.pyc' --exclude .venv
          --exclude node_modules --exclude videos/ --exclude plans/ --exclude design/
          --exclude joe/ --exclude '.DS_Store' --exclude 'wandb/' --exclude 'outputs/')

echo "=== sync rlix: $RLIX_LOCAL -> root@$HOST:$RLIX_REMOTE"
rsync -az $DRY $DELETE "${EXCLUDES[@]}" -e "$RSYNC_SSH" "$RLIX_LOCAL/" "root@$HOST:$RLIX_REMOTE/"

echo "=== sync miles: $MILES_LOCAL -> root@$HOST:$MILES_REMOTE"
rsync -az $DRY $DELETE "${EXCLUDES[@]}" -e "$RSYNC_SSH" "$MILES_LOCAL/" "root@$HOST:$MILES_REMOTE/"

echo "=== sync tooling: scripts/*.{sh,py} -> root@$HOST:/root/"
rsync -az $DRY -e "$RSYNC_SSH" \
  "$RLIX_LOCAL/scripts/vast_setup.sh" \
  "$RLIX_LOCAL/scripts/apply_sglang_patches.py" \
  "$RLIX_LOCAL/scripts/audit_gpu_sampler.sh" \
  "$RLIX_LOCAL/scripts/mock_offload_test.py" \
  "$RLIX_LOCAL/scripts/mock_sglang_offload_test.py" \
  "$RLIX_LOCAL/scripts/test_offload_simple.py" \
  "$RLIX_LOCAL/scripts/vast_debug_run.sh" \
  "$RLIX_LOCAL/scripts/debug_pipeline.py" \
  "root@$HOST:/root/"

if [[ $SETUP -eq 1 && -z "$DRY" ]]; then
  echo "=== running one-shot environment setup on the instance (background, ~10 min)"
  # vast_setup.sh checks out git branches over the synced trees only when they
  # are git repos; on a plain rsync tree it just installs deps + model + data.
  ssh "${SSH_OPTS[@]}" "root@$HOST" \
    'chmod +x /root/vast_setup.sh /root/vast_debug_run.sh /root/audit_gpu_sampler.sh 2>/dev/null;
     setsid nohup bash /root/vast_setup.sh >/dev/null 2>&1 < /dev/null &
     echo "setup launched; follow with: tail -f /root/setup.log (done marker: SETUP_DONE)"'
else
  ssh "${SSH_OPTS[@]}" "root@$HOST" \
    'chmod +x /root/vast_debug_run.sh /root/audit_gpu_sampler.sh 2>/dev/null || true; echo remote-ready'
fi

echo "=== done. debug run: ssh ${SSH_OPTS[*]} root@$HOST 'bash /root/vast_debug_run.sh single'"
