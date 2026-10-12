#!/bin/bash
#SBATCH --job-name=spatial
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/spatial-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/spatial-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-12:00
#SBATCH --cpus-per-task 16
#SBATCH --partition gpu_lowp
#SBATCH --mem=64G
#SBATCH --qos=normal

# One spatial-pipeline script (engine run, merge or synthetic gate), from either tree, on the cluster.
# CPU only: gpu_lowp because the cpu partition is shared with the GLM production queue (event_kernels
# precedent, 2026-10-02); pass --partition=cpu to override. 64 G: the LEC v4 pickle is ~20 GB resident.
# Usage:  sbatch --job-name=<name> sbatch_files/spatial.sbatch <script relative to the repo> [args]
#   e.g.  sbatch --job-name=sp_of_syn sbatch_files/spatial.sbatch code/spatial_tuning_synthetics.py --dataset mec --quick
#         sbatch --job-name=sp_of_bp01 sbatch_files/spatial.sbatch code/run_spatial.py --engine of --dataset mec --recday bp01_20240402_20240403

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, scipy, sklearn, joblib" || { echo "ENV BROKEN"; exit 1; }

SCRIPT="${1:?usage: sbatch spatial.sbatch <script.py> [args]}"
shift || true

python code/check_mirror_parity.py || { echo "MIRROR DRIFT; refusing"; echo "EXIT=1"; exit 1; }

export MPLBACKEND=Agg
# joblib parallelises over shifts; keep BLAS single-threaded so workers do not oversubscribe
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "Running: python $SCRIPT $*"
python "$SCRIPT" "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
