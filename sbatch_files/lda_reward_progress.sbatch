#!/bin/bash
#SBATCH --job-name=lda_rp
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/lda_rp-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/lda_rp-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-12:00
#SBATCH --cpus-per-task 4
#SBATCH --partition cpu
#SBATCH --mem=48G
#SBATCH --qos=normal

# Reward x goal-progress LDA for ONE dataset (all recdays), writing a joinable pickle.
# Usage:  sbatch sbatch_files/lda_reward_progress.sbatch <lec|pfc> [extra args...]
#   e.g.  sbatch --job-name=lda_rp_lec sbatch_files/lda_reward_progress.sbatch lec
# The LEC data_dic is a 3.8 GB monolithic pickle (~20 GB resident); PFC is rebuilt from the
# published per-session files with compute_norm=True (~4 GB). 48 G covers both.

echo "Script started: $(date)"

REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"

module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
echo "Conda environment activated: python=$(which python)"
python -c "import numpy, pandas, sklearn, matplotlib" || { echo "ENV BROKEN"; exit 1; }

DATASET="${1:?usage: sbatch lda_reward_progress.sbatch <lec|pfc> [extra args]}"
shift || true
set -e

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

if [ "$DATASET" = "lec" ]; then SCRIPT=code/run_lda_reward_progress.py;
else SCRIPT=mFC_data/code/run_lda_reward_progress.py; fi

echo "Running reward x goal-progress LDA: dataset=$DATASET extra=$*"
python "$SCRIPT" --dataset "$DATASET" "$@"

echo "All done! $(date)"
