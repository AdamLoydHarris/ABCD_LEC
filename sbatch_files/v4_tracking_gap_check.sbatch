#!/bin/bash
#SBATCH --job-name=v4gapchk
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/%x-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/sbatch_files/slurm_outputs/%x-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-02:00
#SBATCH --cpus-per-task 4
#SBATCH --partition cpu
#SBATCH --mem=64G
#SBATCH --qos=normal

# One data_dic load, no fitting: counts partially-tracked 250 ms rows. See the script docstring.
echo "Script started: $(date)"
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC
module load miniconda
eval "$(conda shell.bash hook)"
conda activate maze_ephys
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS
set +e
python code/v4_tracking_gap_check.py "$@"
rc=$?
echo "All done! $(date) EXIT=$rc"
exit $rc
