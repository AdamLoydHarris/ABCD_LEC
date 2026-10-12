#!/bin/bash
#SBATCH --job-name=resumeks_ly06
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
# maze_ephys_np126 = maze_ephys cloned with numpy pinned to 1.26.4. maze_ephys has numpy
# 2.0.2, which kilosort 4.0.18, spikeinterface 0.101.2 and neo 0.13.3 all forbid
# (pip check); it broke sorting three ways. See docs/handoff/MEC_INTEGRATION.md 8.6.
source activate maze_ephys_np126
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC

python code/resume_kilosort_from_binary.py ly06 2025-07-14_2025-07-15_2025-07-16_2025-07-17_2025-07-18_2025-07-19 9 8
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAILED: python exited $rc at $(date)"
  exit $rc
fi

echo "All done! $(date)"
