#!/bin/bash
#SBATCH --job-name=vjepa2-ssv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # main.py spawns both ranks itself -- do NOT set 2
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=48           # 24 per GPU, matching the known-good teacher job
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --account=torch_pr_230_tandon_priority
#SBATCH --output=/scratch/sd6701/vjepa2_runs/logs/%x_%j.log
#SBATCH --error=/scratch/sd6701/vjepa2_runs/logs/%x_%j.err
#SBATCH --requeue                    # training auto-resumes from latest.pth.tar
#SBATCH --open-mode=append           # a requeued attempt appends to the SAME %j log

set -uo pipefail

# Refuse direct execution: everything below assumes SLURM_* is set.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "this script must be submitted via sbatch (see submit_race.sh), not run directly" >&2
    exit 1
fi

# RUN_ID / CONFIG are exported by submit_race.sh. Checkpoints and logs go to
# <yaml folder>/<RUN_ID>/ (see app/main.py), so every run is isolated; a
# requeue or a resubmission with the SAME RUN_ID resumes from its latest.pth.tar.
: "${RUN_ID:?RUN_ID is not set -- submit via submit_race.sh}"
CONFIG="${CONFIG:-configs/train_2_1/vitG16/pretrain-256px-16f.yaml}"

# -- First-to-start wins (dual-account submission) ----------------------------
# submit_race.sh submits this script twice (torch_pr_230_tandon_priority and
# torch_pr_230_tandon_advanced) WITH THE SAME --job-name; whichever starts
# first scancels the still-pending sibling. If both dispatch in the same
# scheduling cycle, the one with the earlier start time survives (tie -> lower
# job id) -- two live runs would share one RUN_DIR / latest.pth.tar, which
# must never happen.
# NB the block keys off --job-name; submit_race.sh names jobs vjepa2-<RUN_ID>,
# so different runs never see each other as siblings.
# A requeued winner keeps its job id and has no siblings left, so it sails
# through this block.
sibling_jobs () {
    squeue --me --name="${SLURM_JOB_NAME}" -h -o "%i %T %S" | grep -v "^${SLURM_JOB_ID} " || true
}
SIBLINGS=$(sibling_jobs)
if [[ -n "${SIBLINGS}" ]]; then
    PENDING=$(echo "${SIBLINGS}" | awk '$2=="PENDING"{print $1}')
    if [[ -n "${PENDING}" ]]; then
        echo "cancelling pending sibling job(s): ${PENDING}"
        echo "${PENDING}" | xargs -r scancel
    fi
    # Settle, then re-check: simultaneous dispatch means neither sees the other
    # as RUNNING at t=0; after the sleep exactly one of the two defers.
    sleep 15
    MY_START=$(squeue -h -j "${SLURM_JOB_ID}" -o "%S")
    if sibling_jobs | awk -v me="${SLURM_JOB_ID}" -v mystart="${MY_START}" '
        ($2=="RUNNING" || $2=="CONFIGURING") &&
        ($3 < mystart || ($3 == mystart && $1+0 < me+0)) {found=1}
        END {exit !found}'; then
        echo "sibling that started first is already running; deferring and exiting"
        exit 0
    fi
fi

echo "=== winner: job $SLURM_JOB_ID  account=${SLURM_JOB_ACCOUNT:-?}  node=$(hostname) ==="
echo "=== run_id=$RUN_ID  config=$CONFIG ==="

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
  --fname "$CONFIG" \
  --run_id "$RUN_ID" \
  --devices cuda:0 cuda:1

echo "=== finished $(date) ==="
