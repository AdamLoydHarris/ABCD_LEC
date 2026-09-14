#!/bin/bash
# Submit one cross-validated GLM job per recday, then tell you how to merge.
#
#   bash sbatch_files/submit_glm_lec.sh                                  # all recdays, defaults
#   bash sbatch_files/submit_glm_lec.sh --section all_regressors --width-ms 250
#   bash sbatch_files/submit_glm_lec.sh --recdays ah08_20250613_20250615 ly07_...
#
# Recday keys are read from data_dic_lec.pkl so the list stays in sync.
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
    mapfile -t RECDAYS < <(python -c "import pickle; print('\n'.join(str(k) for k in pickle.load(open('data/processed_data/data_dic_lec.pkl','rb')).keys()))")
    if [ "${#RECDAYS[@]}" -eq 0 ]; then
        echo "ERROR: could not read recday keys (env not active? wrong cwd?)." >&2
        echo "  python = $(which python)" >&2
        exit 1
    fi
fi

mkdir -p sbatch_files/slurm_outputs

echo "Submitting ${#RECDAYS[@]} job(s) with args: ${ARGS[*]:-<defaults>}"
for rd in "${RECDAYS[@]}"; do
    jid=$(sbatch --parsable --job-name="glm_${rd}" sbatch_files/glm_lec.sbatch "$rd" "${ARGS[@]:-}")
    echo "  submitted $rd -> job $jid"
done

echo
echo "Monitor with: squeue --me"
echo "When all jobs finish, merge the shards:"
echo "  python code/run_glm_batch.py --merge ${ARGS[*]:-}"
