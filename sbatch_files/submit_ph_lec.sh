#!/bin/bash
# Submit one persistent-(co)homology SLURM job per mouse_recday.
#
#   bash sbatch_files/submit_ph_lec.sh            # all recdays, state=both
#   bash sbatch_files/submit_ph_lec.sh wake       # restrict state
#   bash sbatch_files/submit_ph_lec.sh both ah08_20250613_20250615 ah10_...   # subset
#   TSA_DATA_VERSION=v1 bash sbatch_files/submit_ph_lec.sh                 # the old clock
#
# Recday keys come from the 1 KB session_inds_dic of the version being run, not a 3.8 GB
# unpickle on the login node.

set -euo pipefail
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

STATE="${1:-both}"
shift || true
DATA_VERSION="${TSA_DATA_VERSION:-}"

if [ "$#" -gt 0 ]; then
    RECDAYS=("$@")
else
    module load miniconda >/dev/null 2>&1 || true
    # conda activate via the hook (old `source activate` leaves base active)
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate maze_ephys
    mapfile -t RECDAYS < <(TSA_DV="$DATA_VERSION" python -c "
import os, sys; sys.path.insert(0, 'code')
import recday_registry as rr
print('\n'.join(rr.recdays_for_version(os.environ.get('TSA_DV') or None)))")
    if [ "${#RECDAYS[@]}" -eq 0 ]; then
        echo "ERROR: could not read recday keys (env not active? wrong cwd?)." >&2
        echo "  python = $(which python)" >&2
        exit 1
    fi
fi

echo "Submitting ${#RECDAYS[@]} job(s), state=$STATE, data-version=${DATA_VERSION:-<registry default>}"
for rd in "${RECDAYS[@]}"; do
    jid=$(sbatch --parsable --job-name="ph_${rd}" --export=ALL,TSA_DATA_VERSION="$DATA_VERSION" \
          sbatch_files/ph_lec.sbatch "$rd" "$STATE")
    echo "  submitted $rd -> job $jid"
done
echo "Done. Monitor with: squeue -u \$USER -n ${STATE}  (or squeue --me)"
