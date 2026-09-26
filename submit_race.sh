#!/bin/bash
# Submit the same training job under two accounts/QOS; whichever starts
# first cancels the other (race logic lives in sbatch_train.sh).
#
# Usage:
#   ./submit_race.sh                           # use the two default accounts
#   ./submit_race.sh --list                    # show accounts/QOS you may use
#   ./submit_race.sh SPEC1 SPEC2               # override the accounts
#
# SPEC is  account[:qos]
#
# Every submission is one RUN, identified by RUN_ID. Checkpoints/logs go to
# <yaml folder>/<RUN_ID>/ so runs never overwrite each other.
#   ./submit_race.sh                                  # new run, auto RUN_ID
#   RUN_ID=<id> ./submit_race.sh                      # resume/continue run <id>
set -eu

CONFIG="${CONFIG:-configs/train/vits16/pretrain-256px-16f.yaml}"
[ -f "$CONFIG" ] || { echo "error: config not found: $CONFIG" >&2; exit 1; }
RUN_ID="${RUN_ID:-$(basename "$CONFIG" .yaml)_$(date +%Y%m%d-%H%M%S)}"
# Both siblings share this name (race logic keys off it); unique per run.
JOBNAME="vjepa2-$RUN_ID"

DEFAULT1=torch_pr_230_tandon_priority
DEFAULT2=torch_pr_230_tandon_advanced

if [ "${1:-}" = "--list" ]; then
    echo "=== accounts / partitions / QOS available to $USER ==="
    sacctmgr -n show assoc user="$USER" format=account%30,partition%20,qos%40 2>/dev/null \
        || echo "(sacctmgr unavailable -- try: sshare -U -u $USER)"
    echo
    echo "=== fair-share view ==="
    sshare -U -u "$USER" 2>/dev/null | head -20 || true
    echo
    echo "Then run:  ./submit_race.sh <account[:qos]> <account[:qos]>"
    exit 0
fi

if [ $# -eq 0 ]; then
    set -- "$DEFAULT1" "$DEFAULT2"
elif [ $# -ne 2 ]; then
    echo "error: need 0 or 2 specs (got $#). Run './submit_race.sh --list' to see options." >&2
    exit 1
fi

# Resource flags -- override from the command line without editing files, e.g.
#   PARTITION=gpu GPUFLAG="--gres=gpu:2" ./submit_race.sh
#   GPUFLAG="--gpus-per-node=2" CPUS=32 MEM=128G ./submit_race.sh
GPUFLAG="${GPUFLAG:---gres=gpu:2 --constraint=h100}"
PARTITION="${PARTITION:-}"
CPUS="${CPUS:-48}"
MEM="${MEM:-240G}"
TIME="${TIME:-48:00:00}"

# SLURM will not create the --output directory; it must exist before submit.
mkdir -p /scratch/sd6701/vjepa2_runs/logs

submit_one () {
    local spec="$1"
    local acct="${spec%%:*}"
    local qos=""
    [ "$spec" != "$acct" ] && qos="${spec#*:}"

    local flags=(--parsable --account="$acct" --job-name="$JOBNAME"
                 --export="ALL,RUN_ID=$RUN_ID,CONFIG=$CONFIG"
                 --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME")
    # shellcheck disable=SC2206
    flags+=($GPUFLAG)
    [ -n "$qos" ] && flags+=(--qos="$qos")
    [ -n "$PARTITION" ] && flags+=(--partition="$PARTITION")

    local jid
    if ! jid=$(sbatch "${flags[@]}" sbatch_train.sh); then
        echo "  -> submission FAILED for $spec" >&2
        return 1
    fi
    echo "$jid"
}

if [ -n "$(squeue --me --name="$JOBNAME" -h 2>/dev/null)" ]; then
    echo "error: run $RUN_ID already has jobs queued/running (squeue --me --name=$JOBNAME)" >&2
    exit 1
fi

echo "run_id : $RUN_ID"
echo "config : $CONFIG"
echo "submitting under: $1"
J1=$(submit_one "$1") && echo "  job $J1"
echo "submitting under: $2"
J2=$(submit_one "$2") && echo "  job $J2"

echo
echo "watch  : squeue --me --name=$JOBNAME"
echo "cancel : scancel --me --name=$JOBNAME"
echo "resume : RUN_ID=$RUN_ID CONFIG=$CONFIG ./submit_race.sh"
