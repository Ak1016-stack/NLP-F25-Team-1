#!/usr/bin/env bash
# scripts/remote_deploy.sh
# Usage: ./scripts/remote_deploy.sh <remote_user>@crunchy1.cims.nyu.edu [remote_path]
# Example: ./scripts/remote_deploy.sh ashmit@crunchy1.cims.nyu.edu ~/work/NLP-F25-Team-1

set -euo pipefail
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <remote_user>@host [remote_path]"
  exit 2
fi
REMOTE=$1
REMOTE_PATH=${2:-~/work/NLP-F25-Team-1}
LOCAL_PATH="$(cd "$(dirname "$0")/.." && pwd)"
RSYNC_EXCLUDES=(".git" "outputs" "__pycache__" "*.pyc")

echo "Local repo: $LOCAL_PATH"
echo "Remote: $REMOTE:$REMOTE_PATH"

EXCLUDE_ARGS=()
for e in "${RSYNC_EXCLUDES[@]}"; do
  EXCLUDE_ARGS+=(--exclude="$e")
done

# 1) Sync repo to remote
echo "Syncing repository to remote (rsync)..."
rsync -avP ${EXCLUDE_ARGS[*]} "$LOCAL_PATH/" "$REMOTE:$REMOTE_PATH/"

# 2) Verify remote path exists
echo "Creating remote directory and verifying files..."
ssh "$REMOTE" "mkdir -p $REMOTE_PATH && ls -ld $REMOTE_PATH && ls -la $REMOTE_PATH | head -n 40"

# 3) (Optional) Start an interactive tmux session running training
# This assumes you have conda env named 'hinglish-ner' on remote and the training script is ready.
# Edit the training command below to tune batch_size, epochs, fp16, etc.
TRAIN_CMD="\
cd $REMOTE_PATH && \
source ~/.bashrc || true && \
# activate conda environment (ensure 'hinglish-ner' exists)\
if command -v conda >/dev/null 2>&1; then \
  source $(conda info --base)/etc/profile.d/conda.sh || true; conda activate hinglish-ner || true; \
fi && \
python -m src.train_flat_ner --model_name xlm-roberta-base --output_dir outputs/xlmr-flat-remote --batch_size 8 --num_train_epochs 5 --gradient_accumulation_steps 1"

read -p "Do you want to start training inside a new tmux session on the remote host now? [y/N] " yn
if [[ "$yn" =~ ^[Yy]$ ]]; then
  echo "Starting detached tmux session 'nertrain' on remote to run training command..."
  # create logs dir
  ssh "$REMOTE" "mkdir -p $REMOTE_PATH/logs"
  # start tmux detached and run the command, redirecting stdout/stderr to logs/train-<timestamp>.out
  TS=$(date +%Y%m%d-%H%M%S)
  LOGF="$REMOTE_PATH/logs/train-$TS.out"
  ssh "$REMOTE" "tmux new-session -d -s nertrain 'bash -lc \"$TRAIN_CMD 2>&1 | tee $LOGF\"'"
  echo "Launched tmux session 'nertrain'. To attach: ssh $REMOTE and run 'tmux attach -t nertrain'"
  echo "Remote logs will be written to: $LOGF"
else
  echo "Skipped starting training. You can SSH to the remote host and run the training command manually:" >&2
  echo "  ssh $REMOTE" >&2
  echo "  $TRAIN_CMD" >&2
fi

echo "Done. If you used tmux, attach with: ssh $REMOTE; tmux attach -t nertrain"