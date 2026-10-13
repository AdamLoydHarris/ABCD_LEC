#!/bin/bash
# Submit one event-kernel (M2) job per recday and anchor, then print the merge command.
#
#   bash sbatch_files/submit_event_kernels.sh --dataset lec                       # Stage 1, both anchors
#   bash sbatch_files/submit_event_kernels.sh --dataset pfc --runner drift        # population drift
#   bash sbatch_files/submit_event_kernels.sh --dataset lec --runner stage2 --min-trials 5   # Stage 2
#   bash sbatch_files/submit_event_kernels.sh --dataset lec --min-trials 5        # the sensitivity arm
#   bash sbatch_files/submit_event_kernels.sh --dataset lec --n-perm 20 --recdays ah10_20250616_20250617
#   bash sbatch_files/submit_event_kernels.sh --dataset lec --after 123:124       # start only if those exit 0
#
# Recdays are read from files (LEC full_sessions_v4/full_sessions_*.csv, PFC the latest
# pfc_session_index), never from a pickle. Each job writes one shard, and exits 0 at once if its
# shard already holds the same stamp, so a requeued or resubmitted job costs nothing. Nothing is
# merged until you run the printed command. cpu_backfill: since 2026-10-09 gpu_lowp warns that
# CPU-only jobs will be blocked, and cpu is 260 CPUs at fairshare 0.006.

set -euo pipefail
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda >/dev/null 2>&1 || true
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate maze_ephys

python code/check_mirror_parity.py >/dev/null || { echo "mirror parity FAILED -- not submitting" >&2; exit 1; }

DATASET=""; RUNNER=stage1; MIN_TRIALS=10; N_PERM=100; PARTITION=cpu_backfill
ANCHORS=(entry first_A); RECDAYS=(); AFTER=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --runner) RUNNER="$2"; shift 2 ;;
        --min-trials) MIN_TRIALS="$2"; shift 2 ;;
        --n-perm) N_PERM="$2"; shift 2 ;;
        --partition) PARTITION="$2"; shift 2 ;;
        --after) AFTER="$2"; shift 2 ;;
        --anchors) shift; ANCHORS=(); while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do ANCHORS+=("$1"); shift; done ;;
        --recdays) shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do RECDAYS+=("$1"); shift; done ;;
        *) echo "unknown argument $1" >&2; exit 2 ;;
    esac
done
case "$DATASET" in
    lec) TREE=code ;;
    pfc) TREE=mFC_data/code ;;
    *) echo "--dataset lec|pfc is required" >&2; exit 2 ;;
esac
case "$RUNNER" in
    stage1|drift|stage2) ;;
    *) echo "--runner stage1|drift|stage2" >&2; exit 2 ;;
esac

if [ "${#RECDAYS[@]}" -eq 0 ]; then
    mapfile -t RECDAYS < <(TSA_DS="$DATASET" python - <<'EOF'
import glob, os, re
import pandas as pd
ver = lambda p: int(m.group(1)) if (m := re.search(r'\.v(\d+)\.csv$', p)) else 0
if os.environ['TSA_DS'] == 'lec':
    names = glob.glob('data/processed_data/full_sessions_v4/full_sessions_*.csv')
    rds = {re.match(r'full_sessions_(.+?)(\.v\d+)?\.csv$', os.path.basename(p)).group(1) for p in names}
else:
    p = max(glob.glob('mFC_data/data/processed/pfc_session_index*.csv'), key=ver)
    rds = set(pd.read_csv(p)['recday'])
print('\n'.join(sorted(rds)))
EOF
)
fi
[ "${#RECDAYS[@]}" -gt 0 ] || { echo "no recdays found" >&2; exit 1; }

mkdir -p sbatch_files/slurm_outputs
SB=(sbatch --parsable --partition="$PARTITION")
[ -n "$AFTER" ] && SB+=(--dependency="afterok:$AFTER")
n=0
for rd in "${RECDAYS[@]}"; do
    if [ "$RUNNER" = stage1 ]; then
        for a in "${ANCHORS[@]}"; do
            jid=$("${SB[@]}" --time=0-01:30 --job-name="ek_${DATASET}_mt${MIN_TRIALS}_np${N_PERM}_${a}_${rd}" \
                  sbatch_files/event_kernels.sbatch "$TREE/run_event_kernels.py" --dataset "$DATASET" \
                  --recday "$rd" --anchor "$a" --min-trials "$MIN_TRIALS" --n-perm "$N_PERM")
            echo "  $rd $a -> $jid"
            n=$((n + 1))
        done
    elif [ "$RUNNER" = stage2 ]; then
        # ~8 s per unit (M1 G10); the largest recday-event holds ~150 passers.
        for e in entry A reward; do
            jid=$("${SB[@]}" --time=0-02:00 --job-name="s2_${DATASET}_mt${MIN_TRIALS}_${e}_${rd}" \
                  sbatch_files/event_kernels.sbatch "$TREE/run_event_kernels.py" --dataset "$DATASET" \
                  --stage2 --event "$e" --recday "$rd" --min-trials "$MIN_TRIALS" --n-perm "$N_PERM")
            echo "  $rd $e -> $jid"
            n=$((n + 1))
        done
    else
        jid=$("${SB[@]}" --time=0-00:30 --mem=16G --cpus-per-task=2 --job-name="pd_${DATASET}_mt${MIN_TRIALS}_${rd}" \
              sbatch_files/event_kernels.sbatch "$TREE/run_population_drift.py" --dataset "$DATASET" \
              --recday "$rd" --min-trials "$MIN_TRIALS")
        echo "  $rd -> $jid"
        n=$((n + 1))
    fi
done

echo "submitted $n job(s) on $PARTITION (${#RECDAYS[@]} recdays)"
echo "Read every log to its EXIT= line, then merge:"
if [ "$RUNNER" = stage1 ]; then
    echo "  python $TREE/run_event_kernels.py --dataset $DATASET --merge --min-trials $MIN_TRIALS --n-perm $N_PERM"
elif [ "$RUNNER" = stage2 ]; then
    echo "  python $TREE/run_event_kernels.py --dataset $DATASET --stage2 --merge --min-trials $MIN_TRIALS --n-perm $N_PERM"
else
    echo "  python $TREE/run_population_drift.py --dataset $DATASET --merge --min-trials $MIN_TRIALS"
fi
