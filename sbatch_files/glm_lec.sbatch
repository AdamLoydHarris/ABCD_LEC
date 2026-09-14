#!/bin/bash
#SBATCH --job-name=glm_lec
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/glm-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/glm-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-16:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --qos=normal

# Cross-validated GLM for ONE recday, writing per-recday shards.
# Usage:  sbatch sbatch_files/glm_lec.sbatch <recday> [extra args...]
#   e.g.  sbatch sbatch_files/glm_lec.sbatch ah08_20250613_20250615 --section all_regressors
# Submit all recdays with sbatch_files/submit_glm_lec.sh, then merge with:
#   python code/run_glm_batch.py --merge --section <section> [same config flags]
#
# Each job loads the whole 3.8 GB data_dic for its one recday (~4 min, ~20 GB resident) --
# the pickle is monolithic, so there is no cheaper way in. 64 G and 8 cores mirror the
# persistent-homology jobs.

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
# `conda activate` (via the shell hook), NOT `source activate`, which does not
# honour the user's ~/.conda/envs and silently leaves base active.
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, sklearn" || { echo "ENV BROKEN"; exit 1; }

RECDAY="${1:?usage: sbatch glm_lec.sbatch <recday> [extra args]}"
shift || true
set -e  # fail loudly from here on, after the env is up

# keep BLAS from oversubscribing the 8 cores -- the fit is one big lstsq per fold and
# numpy will otherwise spawn threads that fight each other
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "Running LEC GLM: recday=$RECDAY extra=$*"
python code/run_glm_batch.py "$RECDAY" "$@"

echo "All done! $(date)"
