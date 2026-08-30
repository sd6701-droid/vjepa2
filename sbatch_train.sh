#!/bin/bash
#SBATCH --job-name=vjepa2-ssv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # main.py spawns both ranks itself -- do NOT set 2
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=48           # 24 per GPU, matching the known-good teacher job
#SBATCH --mem=240G
#SBATCH --time=30:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --output=/scratch/sd6701/vjepa2_runs/logs/%x_%j.log
#SBATCH --error=/scratch/sd6701/vjepa2_runs/logs/%x_%j.err
#SBATCH --requeue                    # training auto-resumes from latest.pth.tar
#SBATCH --open-mode=append           # a requeued attempt appends to the SAME %j log

set -uo pipefail

RUN_DIR=/scratch/sd6701/vjepa2_runs/vits.ssv2.256px.16f
LOCK="$RUN_DIR/.race.lock"
mkdir -p "$RUN_DIR"

# ------------------------------------------------------------------ #
# RACE RESOLUTION: the two jobs (one per account) share this job name #
# Whichever starts first wins; the other is cancelled while PENDING.  #
# ------------------------------------------------------------------ #

# 1) Kill any sibling still PENDING. We are RUNNING, so this never hits us.
scancel --me --state=PENDING --name="$SLURM_JOB_NAME" 2>/dev/null || true

# 2) Guard the rare both-start-at-once case. mkdir is atomic.
if ! mkdir "$LOCK" 2>/dev/null; then
    OWNER=$(cat "$LOCK/jobid" 2>/dev/null || echo "")
    if [ "$OWNER" = "$SLURM_JOB_ID" ]; then
        echo "Reclaiming lock after requeue of job $SLURM_JOB_ID."
    elif [ -n "$OWNER" ] && [ -n "$(squeue -h -j "$OWNER" 2>/dev/null)" ]; then
        echo "Job $OWNER already running this experiment. Exiting $SLURM_JOB_ID."
        exit 0
    else
        echo "Stale lock from job ${OWNER:-unknown}; taking over."
    fi
    rm -rf "$LOCK"; mkdir "$LOCK"
fi
echo "$SLURM_JOB_ID" > "$LOCK/jobid"
trap 'rm -rf "$LOCK"' EXIT

echo "=== winner: job $SLURM_JOB_ID  account=${SLURM_JOB_ACCOUNT:-?}  node=$(hostname) ==="

# ---- environment ------------------------------------------------------------
# NOTE: `set -u` must be OFF here. Torch's /etc/bashrc reads $BASHRCSOURCED
# before defining it, and conda's shell hook has the same habit -- both abort
# the job under `set -u`. Re-enabled after activation.
set +u
source ~/.bashrc
conda activate vjepa2
set -u
# -----------------------------------------------------------------------------

set -e
cd /scratch/sd6701/vjepa2          # ADJUST: repo path on Torch

export OMP_NUM_THREADS=8
export WANDB_MODE=offline

echo "--- env check ---"
echo "python : $(which python)"
python -c "import torch; print('torch  :', torch.__version__, '| cuda avail:', torch.cuda.is_available(), '| gpus:', torch.cuda.device_count())"
echo "cwd    : $(pwd)"
nvidia-smi
echo "=== starting $(date) ==="

python -m app.main \
  --fname configs/train_2_1/vitG16/pretrain-256px-16f.yaml \
  --devices cuda:0 cuda:1

echo "=== finished $(date) ==="
