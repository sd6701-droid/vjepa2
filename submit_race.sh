#!/bin/bash
# Submit the same job under two accounts; first to start cancels the other.
# ADJUST these to your real account/partition names -- verify with:
#     sacctmgr show assoc user=$USER format=account,partition,qos
set -eu

ACCT_PRIORITY=torch_priority
ACCT_ADVANCED=torch_advanced

J1=$(sbatch --parsable --account="$ACCT_PRIORITY" sbatch_train.sh)
echo "submitted $J1  (account=$ACCT_PRIORITY)"

J2=$(sbatch --parsable --account="$ACCT_ADVANCED" sbatch_train.sh)
echo "submitted $J2  (account=$ACCT_ADVANCED)"

echo
echo "watch with : squeue --me --name=vjepa2-ssv2"
echo "cancel both: scancel $J1 $J2"
