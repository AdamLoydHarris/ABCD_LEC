#!/bin/bash
# Submit one persistent-(co)homology SLURM job per PFC recday.
#
#   bash sbatch_files/submit_ph_pfc.sh            # all PFC recdays, state=both
#   bash sbatch_files/submit_ph_pfc.sh wake       # restrict state
#   bash sbatch_files/submit_ph_pfc.sh both ah04_01122021_02122021 ...   # subset
#
# Recday keys come from mFC_data/data/MetaData/combined_ABCDonly_days.npy.

set -euo pipefail
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

STATE="${1:-both}"
shift || true

if [ "$#" -gt 0 ]; then
    RECDAYS=("$@")
else
    module load miniconda >/dev/null 2>&1 || true
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate maze_ephys
    mapfile -t RECDAYS < <(python -c "import numpy as np; print('\n'.join(str(x) for x in np.load('mFC_data/data/MetaData/combined_ABCDonly_days.npy', allow_pickle=True)))")
    if [ "${#RECDAYS[@]}" -eq 0 ]; then
        echo "ERROR: could not read PFC recday keys (env not active? wrong cwd?)." >&2
        echo "  python = $(which python)" >&2
        exit 1
    fi
fi

echo "Submitting ${#RECDAYS[@]} PFC job(s), state=$STATE"
for rd in "${RECDAYS[@]}"; do
    jid=$(sbatch --parsable --job-name="phpfc_${rd}" sbatch_files/ph_pfc.sbatch "$rd" "$STATE")
    echo "  submitted $rd -> job $jid"
done
echo "Done. Monitor with: squeue --me"
