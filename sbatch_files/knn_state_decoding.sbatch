#!/bin/bash
#SBATCH --job-name=knn_state
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/knn_state-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/knn_state-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-04:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=48G
#SBATCH --qos=normal

# kNN ABCD state decoding for ONE dataset (all recdays), writing one pickle.
# Usage:  sbatch --job-name=knn_state_lec sbatch_files/knn_state_decoding.sbatch <lec|pfc> [extra]
#   smoke: ... sbatch_files/knn_state_decoding.sbatch lec --recdays ah08_20250613_20250615 \
#                  --out /path/to/smoke.pkl
# The LEC data_dic is a 3.8 GB monolithic pickle (~20 GB resident); PFC is rebuilt with
# compute_norm=True (~4 GB). Synthetic recdays took ~25 s each at a fifth of the null draws.

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, sklearn, scipy" || { echo "ENV BROKEN"; exit 1; }

DATASET="${1:?usage: sbatch knn_state_decoding.sbatch <lec|pfc> [extra args]}"
shift || true
set -e

python code/check_mirror_parity.py || { echo "MIRROR DRIFT; refusing"; exit 1; }

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

if [ "$DATASET" = "lec" ]; then SCRIPT=code/run_knn_state_decoding.py;
else SCRIPT=mFC_data/code/run_knn_state_decoding.py; fi

echo "Running kNN state decoding: dataset=$DATASET extra=$*"
set +e
python -u "$SCRIPT" --dataset "$DATASET" "$@"
rc=$?
set -e

echo "All done! $(date) EXIT=$rc"
exit $rc
