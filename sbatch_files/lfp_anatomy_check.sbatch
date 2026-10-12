#!/bin/bash
#SBATCH --job-name=lfp_anat
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/lfp_anat-%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/lfp_anat-%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-06:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --qos=normal

# LFP anatomy check, one stage per job (code/run_lfp_anatomy_check.py).
# Usage:  sbatch --job-name=lfp_anat_<stage> [--mem=..] sbatch_files/lfp_anatomy_check.sbatch --stage <stage> [args]
#   anatomy  histology env (probe_refit / registration_qc); one brain's volumes at a time, ~32G
#   probemap 26 probe-map configs, 1 GB float32 each + 60 s raw per config for gate A0, ~32G
#   speed    loads data_dic_lec_v4 (3.8 GB pickle, ~20 GB resident), ~48G
#   units    25 recdays of spikes + a process pool for the jitter nulls, ~32G
#   raw      one recording; a 42 min one is 4.9 GB per 384-channel float32 array, ~64G
# LEC-only (El-Gaby's PFC data has no raw continuous data), so nothing here is mirrored.

echo "Script started: $(date) on $(hostname)"
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO/code"

module load miniconda
eval "$(conda shell.bash hook)"
ENV=maze_ephys
for a in "$@"; do [ "$a" = "anatomy" ] && ENV=histology; done
conda activate $ENV
echo "Conda environment $ENV: python=$(which python)"
python -c "import numpy, pandas, scipy" || { echo "ENV BROKEN"; exit 1; }

python check_mirror_parity.py > /dev/null || { echo "MIRROR DRIFT; refusing"; exit 1; }

export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS
export MKL_NUM_THREADS=$OMP_NUM_THREADS

echo "Running: run_lfp_anatomy_check.py $*"
python -u run_lfp_anatomy_check.py "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
