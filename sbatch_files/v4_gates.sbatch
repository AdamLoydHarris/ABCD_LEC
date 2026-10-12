#!/bin/bash
#SBATCH --job-name=v4gates
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-04:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --qos=normal

# Generic runner for the v4-migration Stage 0 gates: pass the script and its args.
#   sbatch sbatch_files/v4_gates.sbatch code/w0_gates.py --data-version v4
echo "Script started: $(date)"
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC
module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
export MPLBACKEND=Agg
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
echo "Running: python -u $*"
set +e
python -u "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
