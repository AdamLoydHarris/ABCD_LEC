#!/bin/bash
#SBATCH --job-name=smoke_ks
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/smoke-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/smoke-%j.err
#SBATCH --nodes 1
#SBATCH --time 0-03:00:00
#SBATCH --cpus-per-task 32
#SBATCH --partition gpu
#SBATCH --gres gpu:1
#SBATCH --mem=200G
#SBATCH --qos=normal

# 20 min of ah08's existing 6-day binary through the SAME code path as production, to prove
# the numpy-2 fixes before spending another 10-24 h per mouse. Two bugs have already cost a
# full run each: probeinterface's repr in probe.prb, and np.NaN in kilosort's swarmsplitter
# (reached only during clustering). Writes to kilosort_output_SMOKE_20min, never the real one.

echo "Script started: $(date)"
module load miniconda
# maze_ephys_np126 = maze_ephys cloned with numpy pinned to 1.26.4. maze_ephys has numpy
# 2.0.2, which kilosort 4.0.18, spikeinterface 0.101.2 and neo 0.13.3 all forbid
# (pip check); it broke sorting three ways. See docs/handoff/MEC_INTEGRATION.md 8.6.
source activate maze_ephys_np126
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC

python code/resume_kilosort_from_binary.py ah08 \
  2025-07-14_2025-07-15_2025-07-16_2025-07-17_2025-07-18_2025-07-19 9 8 --smoke-minutes 20
rc=$?
if [ $rc -ne 0 ]; then echo "FAILED: python exited $rc at $(date)"; exit $rc; fi
echo "All done! $(date)"
