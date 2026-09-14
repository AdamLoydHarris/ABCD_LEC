#!/bin/bash
# Submit ONE V5 run as its own SLURM job.
#
#   sbatch --job-name=v5_pfc_poi_fut --mem=100G --cpus-per-task=12 --time=1-00:00:00 \
#          code/slurm_v5/submit_v5_run.sh 5 12
#
# Why a batch job and not a detached process: the interactive session is itself a SLURM job
# step (3523213) capped at 64 GB, shared with the user's VS Code servers and Jupyter kernels.
# Long fits pushed that cgroup to its limit and were OOM-killed (cgroup memory.events oom_kill
# = 5), which also puts the user's own kernels at risk. A separate job gets its own allocation.
#
# $1 = job id in run_queue_v5.py's ONLY list:
#        1 PFC ElasticNet future   2 PFC Poisson past    5 PFC Poisson future
#        3 LEC Poisson past        4 LEC Poisson future
#      recday-z-scored target (ElasticNet only; `*_zscore_v5_*` dirs, NOT the reproduction):
#        6 PFC ElasticNet past     7 PFC ElasticNet future
#        8 LEC ElasticNet past     9 LEC ElasticNet future     (science config)
#       10 LEC ElasticNet past    11 LEC ElasticNet future     (his criterion, repro_ prefix)
#      LEC Poisson under his criterion + his positive-mean gate (matches PFC specs 2/5):
#       12 LEC Poisson past       13 LEC Poisson future
#      PFC Poisson past regularisation sweep (spec 2's config; glum lasso / elastic net):
#       14 lasso a=0.01   15 lasso a=0.003   16 lasso a=0.001   17 elastic net l1=0.5 a=0.003
# $2 = n_jobs for joblib (defaults to $SLURM_CPUS_PER_TASK)
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/slurm_v5/logs/%x_%j.out
#SBATCH --error=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC/code/slurm_v5/logs/%x_%j.out
set -euo pipefail
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
PY=/nfs/nhome/live/aharris/.conda/envs/maze_ephys/bin/python3
JOB=${1:?job id required (1,2,3,4,5)}
NJ=${2:-${SLURM_CPUS_PER_TASK:-6}}
mkdir -p "$REPO/code/slurm_v5/logs"
cd "$REPO/code/slurm_v5"
echo "host $(hostname) | slurm job ${SLURM_JOB_ID:-none} | mem ${SLURM_MEM_PER_NODE:-?}MB | cpus ${SLURM_CPUS_PER_TASK:-?} | n_jobs $NJ | run spec $JOB"
srun "$PY" run_queue_v5.py "$NJ" "$JOB"
