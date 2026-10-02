#!/bin/bash
#SBATCH --job-name=event_kernels
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/event_kernels-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/event_kernels-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-04:00
#SBATCH --cpus-per-task 8
#SBATCH --partition gpu_lowp
#SBATCH --mem=32G
#SBATCH --qos=normal

# One event-kernel script, from either tree, on the cluster. CPU only: gpu_lowp because the cpu
# partition was degraded on 2026-10-02 (user-approved); pass --partition=cpu to override.
# Usage:  sbatch --job-name=<name> sbatch_files/event_kernels.sbatch <script relative to the repo> [args]
#   e.g.  sbatch --job-name=ek_syn_lec sbatch_files/event_kernels.sbatch code/event_kernels_synthetics.py --dataset lec --quick

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, scipy, glum" || { echo "ENV BROKEN"; exit 1; }

SCRIPT="${1:?usage: sbatch event_kernels.sbatch <script.py> [args]}"
shift || true

python code/check_mirror_parity.py || { echo "MIRROR DRIFT; refusing"; exit 1; }

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

echo "Running: python $SCRIPT $*"
python "$SCRIPT" "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
