#!/bin/bash
#SBATCH --job-name=hist_stage
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/hist-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/hist-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-06:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=32G
#SBATCH --qos=normal

# The histology-env twin of mec_stage.sbatch: registration QC, probe tools, anything needing
# brainreg / brainglobe / skimage. Usage:
#   sbatch sbatch_files/histology_stage.sbatch <script.py> [args...]
# The script name is relative to code/histology_refit/ (absolute paths also work).
# BRAINGLOBE_CONFIG_DIR points the env at the shared atlas cache (it has allen_mouse_10um; the
# per-user ~/.brainglobe only has the 25 um atlas). 32 G: the QC reads page ranges, never a whole
# volume; the synthetic gate's verify_contacts transient is 4.3 GB.

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO/code/histology_refit"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate histology
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, scipy, tifffile, skimage, brainglobe_atlasapi" || { echo "ENV BROKEN"; exit 1; }

export BRAINGLOBE_CONFIG_DIR=/ceph/apps/application_data/brainglobe
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

SCRIPT="${1:?usage: sbatch histology_stage.sbatch <script.py> [args]}"
shift || true

echo "Running: python $SCRIPT $*"
python "$SCRIPT" "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
