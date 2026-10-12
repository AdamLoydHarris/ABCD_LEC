#!/bin/bash
#SBATCH --job-name=qconly_ly05
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/qconly-%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/preprocessing/slurm_outputs/qconly-%j.err
#SBATCH --nodes 1
#SBATCH --time 2-00:00:00
#SBATCH --cpus-per-task 8
#SBATCH --partition cpu
#SBATCH --mem=96G

# ly05's 6-day sort (3666093) finished Kilosort -- 430 units -- then hit the 7-day walltime
# 5.9 h into quality metrics; that stage took 3.9-6.2 h on the other four blocks. Every analyzer
# extension except quality_metrics is saved, and get_quality_metrics reuses a saved analyzer, so
# this reads the finished sort and runs only the metrics and the gate. CPU only: no GPU needed,
# and compute_quality_metrics runs single-threaded here (n_jobs unset), so 8 cores is plenty.

echo "Script started: $(date)"
module load miniconda
source activate maze_ephys_np126
cd /ceph/behrens/adam_harris/Taskspace_abstraction_lEC

python code/resume_kilosort_from_binary.py ly05 \
  2025-07-14_2025-07-15_2025-07-16_2025-07-17_2025-07-18_2025-07-19 9 8 --qc-only
rc=$?
if [ $rc -ne 0 ]; then echo "FAILED: python exited $rc at $(date)"; exit $rc; fi
echo "All done! $(date)"
