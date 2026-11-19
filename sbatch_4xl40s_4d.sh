#!/bin/bash
#SBATCH --job-name=vjepa2-ssv2-300ep-t2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # app.main spawns one process per GPU itself
#SBATCH --partition=gl40s_long         # gl40s_short caps at 3 days; long allows up to 28
#SBATCH --gres=gpu:4                 # 4x L40S -> global batch 4 x 32 = 128
#SBATCH --cpus-per-task=32           # 8 dataloader workers per GPU
#SBATCH --mem=200G
#SBATCH --time=4-00:00:00            # 4 days; 300 t2 epochs need ~5 -> resubmit with RUN_ID to finish
#SBATCH --output=/gpfs/scratch/sd6701/vjepa2_runs/logs/%x_%j.log
#SBATCH --error=/gpfs/scratch/sd6701/vjepa2_runs/logs/%x_%j.err
#SBATCH --requeue                    # a requeue keeps the job id -> same RUN_ID -> resumes from latest.pt
#SBATCH --open-mode=append
#
# Submit (logs dir must exist first; SLURM does not create it):
#   mkdir -p /gpfs/scratch/sd6701/vjepa2_runs/logs
#   sbatch sbatch_4xl40s_4d.sh
# Resume a run after the time limit (same checkpoints / wandb run):
#   sbatch --export=ALL,RUN_ID=<id from the log> sbatch_4xl40s_4d.sh
#   (keep 4 GPUs when resuming so steps/epoch and the LR schedule stay the same)
# Other configs:
#   sbatch --export=ALL,CONFIG=configs/train/vits16/<file>.yaml sbatch_4xl40s_4d.sh

set -uo pipefail

REPO=/gpfs/scratch/sd6701/personal/salt/vjepa2
ENV=/gpfs/scratch/sd6701/conda-envs/vjepa2-312
CONFIG="${CONFIG:-configs/train/vits16/pretrain-256px-16f-300ep-ssv2.yaml}"
# Default id is tied to the SLURM job id, so it is stable across requeues.
RUN_ID="${RUN_ID:-$(basename "$CONFIG" .yaml)_${SLURM_JOB_ID}}"

# conda's shell hook trips over `set -u`
set +u
source ~/.bashrc
conda activate "$ENV" || { echo "conda activate $ENV failed" >&2; exit 1; }
set -u
set -e

cd "$REPO"
export OMP_NUM_THREADS=8
export WANDB_MODE="${WANDB_MODE:-online}"   # use WANDB_MODE=offline if compute nodes have no internet

NGPU=$(python -c "import torch; print(torch.cuda.device_count())")
DEVICES=$(seq -s ' ' -f 'cuda:%g' 0 $((NGPU - 1)))

echo "=== job $SLURM_JOB_ID on $(hostname) | run_id=$RUN_ID | config=$CONFIG ==="
echo "python: $(which python) | gpus: $NGPU -> $DEVICES"
nvidia-smi
echo "=== starting $(date) ==="

# shellcheck disable=SC2086
python -m app.main \
  --fname "$CONFIG" \
  --run_id "$RUN_ID" \
  --devices $DEVICES

echo "=== finished $(date) ==="
