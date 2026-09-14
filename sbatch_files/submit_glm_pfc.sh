#!/bin/bash
# Submit one cross-validated GLM job per PFC recday, then tell you how to merge.
#
#   bash sbatch_files/submit_glm_pfc.sh                                  # all recdays, defaults
#   bash sbatch_files/submit_glm_pfc.sh --section all_regressors --width-ms 250
#   bash sbatch_files/submit_glm_pfc.sh --recdays ah08_20250613_20250615 ly07_...
#
# Recday keys are read from MetaData/combined_ABCDonly_days.npy (the canonical 25
# double-day ABCD recordings) so the list stays in sync.
#
# Each job writes its own shard; nothing writes the section-level pickle until you run the
# merge printed at the end. That is deliberate -- 25 jobs writing one pickle would race, and
# the loser's recdays would vanish silently (the file still loads, just with fewer keys).

set -euo pipefail
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda >/dev/null 2>&1 || true
# conda activate via the hook (old `source activate` leaves base active)
eval "$(conda shell.bash hook)" 2>/dev/null || true
conda activate maze_ephys

# Refuse to launch if code/ and mFC_data/code/ have drifted. Two 25-job runs have already died
# on a mirror lag (a missing import, then glm_cv.py 150 lines behind); this catches both.
python code/check_mirror_parity.py || { echo "mirror parity FAILED -- not submitting" >&2; exit 1; }

RECDAYS=()
ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --recdays) shift; while [ "$#" -gt 0 ] && [[ "$1" != --* ]]; do RECDAYS+=("$1"); shift; done ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

if [ "${#RECDAYS[@]}" -eq 0 ]; then
    mapfile -t RECDAYS < <(python -c "import numpy as np, os; print('\n'.join(str(r) for r in np.load(os.path.join('mFC_data','data','MetaData','combined_ABCDonly_days.npy'))))")
    if [ "${#RECDAYS[@]}" -eq 0 ]; then
        echo "ERROR: could not read recday keys (env not active? wrong cwd?)." >&2
        echo "  python = $(which python)" >&2
        exit 1
    fi
fi

mkdir -p sbatch_files/slurm_outputs

echo "Submitting ${#RECDAYS[@]} job(s) with args: ${ARGS[*]:-<defaults>}"
for rd in "${RECDAYS[@]}"; do
    jid=$(sbatch --parsable --job-name="glm_${rd}" sbatch_files/glm_pfc.sbatch "$rd" "${ARGS[@]:-}")
    echo "  submitted $rd -> job $jid"
done

echo
echo "Monitor with: squeue --me"
echo "When all jobs finish, merge the shards:"
echo "  python mFC_data/code/run_glm_batch.py --merge ${ARGS[*]:-}"
