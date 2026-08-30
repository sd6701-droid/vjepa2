#!/bin/bash
#SBATCH --job-name=vjepa2-ssv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # main.py spawns the 2 ranks itself -- do NOT set 2
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=64           # 32 per rank x 2 ranks
#SBATCH --mem=256G
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/sd6701/vjepa2_runs/logs/%x-%A-%j.out
#SBATCH --error=/scratch/sd6701/vjepa2_runs/logs/%x-%A-%j.err

set -uo pipefail

RUN_DIR=/scratch/sd6701/vjepa2_runs/vits.ssv2.256px.16f
LOCK="$RUN_DIR/.race.lock"
mkdir -p /scratch/sd6701/vjepa2_runs/logs "$RUN_DIR"

# ------------------------------------------------------------------ #
# RACE RESOLUTION: two jobs submitted under different accounts share  #
# this job name. Whichever starts first wins; the other is cancelled. #
# ------------------------------------------------------------------ #

# 1) Kill any sibling still PENDING. We are RUNNING, so this never hits us.
scancel --me --state=PENDING --name="$SLURM_JOB_NAME" 2>/dev/null || true

# 2) Guard the rare case where BOTH start at the same instant: mkdir is
#    atomic, so exactly one job can create the lock and proceed.
if ! mkdir "$LOCK" 2>/dev/null; then
    OWNER=$(cat "$LOCK/jobid" 2>/dev/null || echo "")
    # Stale lock (owner no longer in the queue)? Steal it.
    if [ -n "$OWNER" ] && squeue -h -j "$OWNER" >/dev/null 2>&1 && [ -n "$(squeue -h -j "$OWNER" 2>/dev/null)" ]; then
        echo "Job $OWNER already running this experiment. Exiting $SLURM_JOB_ID."
        exit 0
    fi
    echo "Stale lock from job ${OWNER:-unknown}; taking over."
    rm -rf "$LOCK" && mkdir "$LOCK"
fi
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
trap 'rm -rf "$LOCK"' EXIT

echo "=== winner: job $SLURM_JOB_ID  account=${SLURM_JOB_ACCOUNT:-?}  partition=${SLURM_JOB_PARTITION:-?} ==="

set -e
# ---- environment (ADJUST to your setup) -------------------------------------
module purge
source ~/.bashrc
conda activate vjepa2
# -----------------------------------------------------------------------------

cd /scratch/sd6701/vjepa2          # ADJUST: repo path on the cluster

export OMP_NUM_THREADS=8
export WANDB_MODE=offline          # compute nodes typically have no egress

nvidia-smi
echo "=== starting $(date) ==="

python -m app.main \
  --fname configs/train_2_1/vitG16/pretrain-256px-16f.yaml \
  --devices cuda:0 cuda:1

echo "=== finished $(date) ==="
