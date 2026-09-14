#!/bin/bash
#SBATCH --job-name=ph_lec
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/ph-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/ph-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --qos=normal

# Persistent (co)homology sweep for ONE mouse_recday (wake + sleep).
# Usage:  sbatch sbatch_files/ph_lec.sbatch <mouse_recday> [state] [extra args...]
#   e.g.  sbatch sbatch_files/ph_lec.sbatch ah08_20250613_20250615
#         sbatch sbatch_files/ph_lec.sbatch ah08_20250613_20250615 sleep
# Submit all 24 recdays with sbatch_files/submit_ph_lec.sh

# NB: do NOT `set -u` before conda activation — conda's activate scripts
# reference unbound shell vars and silently fail to set PATH under `set -u`,
# leaving the base python (no numpy) active. Plain bash, like the repo's
# merge_*.sbatch jobs.
echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
# `conda activate` (via the shell hook), NOT the old `source activate`, which
# does not honour the user's ~/.conda/envs and silently leaves base active.
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, ripser, dreimac" || { echo "ENV BROKEN"; exit 1; }

RECDAY="${1:?usage: sbatch ph_lec.sbatch <mouse_recday> [state] [extra args]}"
STATE="${2:-both}"
shift || true
shift || true
set -e  # fail loudly from here on, after the env is up

# keep BLAS from oversubscribing the 8 cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "Running persistent (co)homology: recday=$RECDAY state=$STATE extra=$*"
python code/run_ph_batch.py "$RECDAY" "$STATE" "$@"

echo "All done! $(date)"
