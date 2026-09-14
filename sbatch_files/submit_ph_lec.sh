#!/bin/bash
# Submit one persistent-(co)homology SLURM job per mouse_recday.
#
#   bash sbatch_files/submit_ph_lec.sh            # all recdays, state=both
#   bash sbatch_files/submit_ph_lec.sh wake       # restrict state
#   bash sbatch_files/submit_ph_lec.sh both ah08_20250613_20250615 ah10_...   # subset
#
# Recday keys are read from data_dic_lec.pkl so the list stays in sync.

set -euo pipefail
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

STATE="${1:-both}"
shift || true

if [ "$#" -gt 0 ]; then
    RECDAYS=("$@")
else
    module load miniconda >/dev/null 2>&1 || true
    # conda activate via the hook (old `source activate` leaves base active)
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate maze_ephys
    mapfile -t RECDAYS < <(python -c "import pickle; print('\n'.join(str(k) for k in pickle.load(open('data/processed_data/data_dic_lec.pkl','rb')).keys()))")
    if [ "${#RECDAYS[@]}" -eq 0 ]; then
        echo "ERROR: could not read recday keys (env not active? wrong cwd?)." >&2
        echo "  python = $(which python)" >&2
        exit 1
    fi
fi

echo "Submitting ${#RECDAYS[@]} job(s), state=$STATE"
for rd in "${RECDAYS[@]}"; do
    jid=$(sbatch --parsable --job-name="ph_${rd}" sbatch_files/ph_lec.sbatch "$rd" "$STATE")
    echo "  submitted $rd -> job $jid"
done
echo "Done. Monitor with: squeue -u \$USER -n ${STATE}  (or squeue --me)"
