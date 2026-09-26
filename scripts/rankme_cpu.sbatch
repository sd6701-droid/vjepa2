#!/bin/bash
# RankMe for every 10th checkpoint of a V-JEPA run, on a CPU compute node, logged to wandb.
#
# Submit from the repo root:
#     sbatch scripts/rankme_cpu.sbatch
#     sbatch scripts/rankme_cpu.sbatch /path/to/other/run_dir      # different run
#
# Uncomment/set --account (and --partition) if your jobs normally need them.
#SBATCH --job-name=rankme
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=rankme_%j.out
##SBATCH --account=<your_account>
##SBATCH --partition=<cpu_partition>

set -euo pipefail

RUN_DIR=${1:-/scratch/sd6701/vjepa2_runs/vits.ssv2.256px.16f.v2/interactive_20260926-052920}
DATA=/scratch/sd6701/datasets/ssv2/ssv2_val_labeled.csv
CACHE=/scratch/sd6701/rankme_ssv2_512clips.pt

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vjepa2-312

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# compute nodes without internet: uncomment, then run `wandb sync <run_dir>/wandb/offline-run-*` on a login node
# export WANDB_MODE=offline

echo "node: $(hostname)  cpus: $SLURM_CPUS_PER_TASK  run: $RUN_DIR"
python scripts/rankme.py "$RUN_DIR" \
    --every 10 \
    --data "$DATA" \
    --device cpu \
    --num-samples 512 \
    --batch-size 8 \
    --num-workers 8 \
    --cache "$CACHE" \
    --wandb
