#!/bin/bash
#SBATCH --job-name=resumeks_ah08
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/resumeks-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/resumeks-%j.err
#SBATCH --nodes 1
#SBATCH --time 5-00:00:00
#SBATCH --cpus-per-task 32
#SBATCH --partition gpu
#SBATCH --gres gpu:a100:1
#SBATCH --mem=200G
#SBATCH --qos=normal

# GPU pinned to a100 (39.49 GB). Final clustering of a 6-day block needs ~20.4 GB of
# VRAM: ah10 died there on a 19.57 GB A4500 (3666092, torch.OutOfMemoryError after 37 h),
# and ah08 was cancelled off a 15.88 GB P5000. Only a100 and l40s clear it; no node
# features exist to constrain on size, so the type is named.
# mem 200G: Kilosort's own peak was 62.97 GB; the 471.8 GB MaxRSS is the memmapped 1.6 TB
# binary filling the 450G cgroup ceiling exactly (450 GiB = 471 859 200 K), i.e. page cache.
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

python code/resume_kilosort_from_binary.py ah08 2025-07-14_2025-07-15_2025-07-16_2025-07-17_2025-07-18_2025-07-19 9 8
rc=$?
if [ $rc -ne 0 ]; then
  echo "FAILED: python exited $rc at $(date)"
  exit $rc
fi

echo "All done! $(date)"
