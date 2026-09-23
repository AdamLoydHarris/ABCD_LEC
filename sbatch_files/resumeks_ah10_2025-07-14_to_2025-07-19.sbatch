#!/bin/bash
#SBATCH --job-name=resumeks_ah10
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/resumeks-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/resumeks-%j.err
#SBATCH --nodes 1
#SBATCH --time 5-00:00:00
#SBATCH --cpus-per-task 32
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem=450G
#SBATCH --qos=normal

# Resume: the destriped binary/ already exists (1.6 TB, verified sample-exact against
# recording_sessions_in_concat.csv). The first attempt died in run_sorter on the numpy-2
# probe-file bug, now fixed in the merge script. This skips the ~1.5 h destriping and the
# 1.6 TB rewrite and runs Kilosort onward, reusing the merge script's own QC functions.

echo "Script started: $(date)"
module load miniconda
source activate maze_ephys
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC

python code/resume_kilosort_from_binary.py ah10 2025-07-14_2025-07-15_2025-07-16_2025-07-17_2025-07-18_2025-07-19 9 8
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAILED: python exited $rc at $(date)"
  exit $rc
fi

echo "All done! $(date)"
