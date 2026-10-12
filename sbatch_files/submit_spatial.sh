#!/bin/bash
# One spatial-pipeline job per recday, all under the same run tag, then print the merge command.
# Usage:  bash sbatch_files/submit_spatial.sh <engine: of|maze> <dataset: mec|lec|pfc> <recday> [<recday> ...]
# The tag is the CLI default (hash of config + code), so it changes whenever the recipe does.
set -euo pipefail
ENGINE=$1; DATASET=$2; shift 2
REPO=/ceph/behrens/adam_harris/Taskspace_abstraction_lEC
cd "$REPO"
CODE=code
[ "$DATASET" = pfc ] && CODE=mFC_data/code
PY=/nfs/nhome/live/aharris/.conda/envs/maze_ephys/bin/python
# SPATIAL_CONFIG, a JSON of overrides (e.g. a grid-annulus sensitivity run), is hashed into the tag
CFG=()
[ -n "${SPATIAL_CONFIG:-}" ] && CFG=(--config "$SPATIAL_CONFIG")
python_tag=$($PY $CODE/run_spatial.py --engine "$ENGINE" --dataset "$DATASET" ${CFG[@]+"${CFG[@]}"} --print-tag)
echo "run tag $python_tag"
for RD in "$@"; do
  # SBATCH_EXTRA, e.g. "--time=1-00:00 --cpus-per-task=32", passes through to every job
  sbatch --job-name="sp_${ENGINE}_${RD}" --partition=cpu_backfill ${SBATCH_EXTRA:-} sbatch_files/spatial.sbatch \
    "$CODE/run_spatial.py" --engine "$ENGINE" --dataset "$DATASET" --recday "$RD" --run-tag "$python_tag" ${CFG[@]+"${CFG[@]}"}
done
echo "merge when done:  $PY $CODE/run_spatial.py --engine $ENGINE --dataset $DATASET --merge --expect-recdays $# --run-tag $python_tag ${CFG[@]+"${CFG[@]}"}"
