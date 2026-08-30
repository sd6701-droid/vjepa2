#!/bin/bash
#SBATCH --job-name=vjepa2-ssv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1          # main.py spawns the 2 ranks itself -- do NOT set 2
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=64           # 32 per rank x 2 ranks
#SBATCH --mem=256G
#SBATCH --time=40:00:00
#SBATCH --output=/scratch/sd6701/vjepa2_runs/logs/%x-%j.out
#SBATCH --error=/scratch/sd6701/vjepa2_runs/logs/%x-%j.err

set -euo pipefail

mkdir -p /scratch/sd6701/vjepa2_runs/logs

# ---- environment (ADJUST to your setup) -------------------------------------
module purge
# module load cuda/12.1.1
source ~/.bashrc
conda activate vjepa2
# -----------------------------------------------------------------------------

cd /scratch/sd6701/vjepa2          # ADJUST: path to the repo on the cluster

export OMP_NUM_THREADS=8
export WANDB_MODE=offline          # Greene compute nodes have no outbound internet

CONFIG=configs/train_2_1/vitG16/pretrain-256px-16f.yaml

nvidia-smi
echo "=== starting $(date) ==="

python -m app.main \
  --fname "$CONFIG" \
  --devices cuda:0 cuda:1

echo "=== finished $(date) ==="
