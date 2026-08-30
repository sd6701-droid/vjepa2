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
set -eu

JOBNAME=vjepa2-ssv2

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

submit_one () {
    local spec="$1"
    local acct="${spec%%:*}"
    local qos=""
    [ "$spec" != "$acct" ] && qos="${spec#*:}"

    local flags=(--parsable --account="$acct")
    [ -n "$qos" ] && flags+=(--qos="$qos")

    local jid
    if ! jid=$(sbatch "${flags[@]}" sbatch_train.sh); then
        echo "  -> submission FAILED for $spec" >&2
        return 1
    fi
    echo "$jid"
}

echo "submitting under: $1"
J1=$(submit_one "$1") && echo "  job $J1"
echo "submitting under: $2"
J2=$(submit_one "$2") && echo "  job $J2"

echo
echo "watch  : squeue --me --name=$JOBNAME"
echo "cancel : scancel --me --name=$JOBNAME"
