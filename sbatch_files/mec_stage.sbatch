#!/bin/bash
#SBATCH --job-name=mec_stage
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/mec-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/mec-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-06:00
#SBATCH --cpus-per-task 4
#SBATCH --partition cpu
#SBATCH --mem=48G
#SBATCH --qos=normal

# One preprocessing stage of the mEC port, on the cluster rather than the interactive cgroup.
# Usage:  sbatch sbatch_files/mec_stage.sbatch <script.py> [args...]
#   e.g.  sbatch sbatch_files/mec_stage.sbatch extract_neuron_raw.py --dataset mec --write
#         sbatch sbatch_files/mec_stage.sbatch build_data_dic.py --dataset mec
# The script name is relative to code/preprocessing/. `--dataset mec` is passed through,
# not implied, so the log records it.
#
# 48 G: extract_neuron_raw loads one block's spike_times (~35 M int64) plus the QC table;
# build_data_dic holds the whole mEC data_dic (251 merged + 606 single units) in memory.

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO/code/preprocessing"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, scipy, h5py" || { echo "ENV BROKEN"; exit 1; }

SCRIPT="${1:?usage: sbatch mec_stage.sbatch <script.py> [args]}"
shift || true
set -e

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1   # progress lines reach the .out as they happen

echo "Running: python $SCRIPT $*"
python "$SCRIPT" "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
