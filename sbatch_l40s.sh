#!/bin/bash
#SBATCH --job-name=vjepa2-mix-20ep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # app.main spawns one process per GPU itself
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:3                 # 3x L40S (use gpu:l40s:3 if the cluster names GPU types)
#SBATCH --cpus-per-task=24           # 8 dataloader workers per GPU
#SBATCH --mem=160G
#SBATCH --time=2-12:00:00            # ~50 h expected for 20 mix epochs on 3x L40S
#SBATCH --output=/gpfs/scratch/sd6701/vjepa2_runs/logs/%x_%j.log
#SBATCH --error=/gpfs/scratch/sd6701/vjepa2_runs/logs/%x_%j.err
#SBATCH --requeue                    # a requeue keeps the job id -> same RUN_ID -> resumes from latest.pt
#SBATCH --open-mode=append
#
# Submit (logs dir must exist first; SLURM does not create it):
#   mkdir -p /gpfs/scratch/sd6701/vjepa2_runs/logs
#   sbatch sbatch_l40s.sh
# Resume a run after the time limit (same checkpoints / wandb run):
#   RUN_ID=<id from the log> sbatch --export=ALL,RUN_ID=<id> sbatch_l40s.sh
# Other configs:
#   sbatch --export=ALL,CONFIG=configs/train/vits16/<file>.yaml sbatch_l40s.sh

set -uo pipefail

REPO=/gpfs/scratch/sd6701/personal/salt/vjepa2
ENV=/gpfs/scratch/sd6701/conda-envs/vjepa2-312
CONFIG="${CONFIG:-configs/train/vits16/pretrain-256px-16f-20ep-mix.yaml}"
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
