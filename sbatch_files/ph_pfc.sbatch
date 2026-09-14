#!/bin/bash
#SBATCH --job-name=ph_pfc
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/ph-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/ph-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 1-00:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --qos=normal

# Persistent (co)homology sweep for ONE PFC recday (wake + sleep).
# Usage:  sbatch sbatch_files/ph_pfc.sbatch <recday> [state] [extra args...]
#   e.g.  sbatch sbatch_files/ph_pfc.sbatch ah04_01122021_02122021
# Submit all PFC recdays with sbatch_files/submit_ph_pfc.sh

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
# `conda activate` (via the shell hook), NOT `source activate`, which does not
# honour the user's ~/.conda/envs and silently leaves base active.
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, ripser, dreimac, joblib" || { echo "ENV BROKEN"; exit 1; }

RECDAY="${1:?usage: sbatch ph_pfc.sbatch <recday> [state] [extra args]}"
STATE="${2:-both}"
shift || true
shift || true
set -e  # fail loudly from here on, after the env is up

# keep BLAS from oversubscribing the 8 cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "Running PFC persistent (co)homology: recday=$RECDAY state=$STATE extra=$*"
python mFC_data/code/run_ph_batch.py "$RECDAY" "$STATE" "$@"

echo "All done! $(date)"
